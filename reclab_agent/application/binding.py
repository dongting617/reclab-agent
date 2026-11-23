"""Bind task configuration to the common experiment tool and Agent strategy."""
from __future__ import annotations
import os
from pathlib import Path
import uuid
from reclab_agent.execution.config import load_policy
from reclab_agent.core import OpenAIResponsesBackend
from reclab_agent.execution.worker import DockerWorker, WorkerPolicy
from reclab_agent.infrastructure.files import read_regular_file
from reclab_agent.tools.remote import RemoteTools
from reclab_agent.execution.ssh import SSHConnection
from reclab_agent.tools import RecLabToolbox
from reclab_agent.experiments.workspace import create_workspace
from reclab_agent.domain.configuration import (
    TaskConfiguration,
    load_task_configuration,
    read_json_strict,
)
from reclab_agent.domain.research import PlanReview, ResearchPlan
from reclab_agent.experiments.adapters import TaskAdapterRegistry
from reclab_agent.application.inspection import inspect_task
from reclab_agent.agents.optimization import OptimizationAgent
from reclab_agent.experiments.runner import ExperimentRunner
from reclab_agent.experiments.remote_runner import RemoteExperimentRunner
from reclab_agent.tools.toolbox import OptimizationToolbox


def bind_task_tools(configuration: TaskConfiguration | str | Path, *, plan: ResearchPlan,
                    review: PlanReview | None, policy: WorkerPolicy | None = None,
                    registry: TaskAdapterRegistry | None = None,
                    worker_factory=DockerWorker, remote_tools=None) -> OptimizationToolbox:
    """Configure the standard tool once; target models never supply a runner.

    A trusted controller supplies the matching plan/review and execution policy.
    No review is invented from JSON and this function does not launch jobs.
    """
    config = (configuration if isinstance(configuration, TaskConfiguration) else
              load_task_configuration(configuration))
    registry = registry if registry is not None else TaskAdapterRegistry()
    inspection = inspect_task(config, registry=registry)
    if not inspection["inputs_valid"]:
        details = "; ".join(f"{error['check']}: {error['detail']}" for error in inspection["errors"])
        raise ValueError(f"task input checks failed: {details}")
    if config.execution_backend == "ssh_tmux":
        if policy is not None:
            raise ValueError("the ssh_tmux profile does not use a Docker policy")
        if remote_tools is None:
            if config.execution_connection_path is None:
                raise ValueError("configure an SSH connection before binding the experiment tool")
            connection = SSHConnection.from_json(config.execution_connection_path)
            remote_tools = RemoteTools(connection, upload_roots={
                "source": config.output_directory / "snapshots", "public": config.dataset_directory / "public",
            }, download_root=config.output_directory)
        runner = RemoteExperimentRunner(config, plan, remote_tools, registry=registry)
    else:
        if policy is None:
            if config.execution_policy_path is None:
                raise ValueError("configure a trusted execution policy before binding the experiment tool")
            policy = load_policy(config.execution_policy_path)
        runner = ExperimentRunner(config, plan, policy, registry=registry, worker_factory=worker_factory)
    files = {name: read_regular_file(config.source_directory / name, max_bytes=1_000_000)
             .decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
             for name in inspection["source"]["files"]}
    workspace = create_workspace(config.output_directory / "workspaces", "candidate-" + uuid.uuid4().hex[:12],
                                 files, editable_paths=list(config.editable_files))
    return OptimizationToolbox(workspace, task=config.task, plan=plan, review=review,
                           snapshots_root=config.output_directory / "snapshots", run_experiment=runner)


def create_task_agent(configuration: TaskConfiguration | str | Path, *, plan: ResearchPlan,
                      review: PlanReview | None, backend=None, policy: WorkerPolicy | None = None,
                      registry: TaskAdapterRegistry | None = None,
                      worker_factory=DockerWorker, remote_tools=None) -> OptimizationAgent:
    """Construct an Agent with the shared tool; call run() to start its decisions.

    backend/worker_factory injection is for trusted host integration and tests,
    not fields that a target model or Agent can supply through its task JSON.
    """
    config = (configuration if isinstance(configuration, TaskConfiguration) else
              load_task_configuration(configuration))
    if backend is None:
        model = os.environ.get(config.model_env)
        key = os.environ.get(config.api_key_env)
        if not model or not key:
            raise ValueError("configure the Agent model and credentials in the named environment variables")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install the optional OpenAI backend before creating an online Agent") from error
        backend = OpenAIResponsesBackend(model, client=OpenAI(api_key=key))
    research = bind_task_tools(config, plan=plan, review=review, policy=policy,
                               registry=registry, worker_factory=worker_factory, remote_tools=remote_tools)
    selected = read_json_strict(config.model_config_path, max_bytes=1_000_000)["model"]
    return OptimizationAgent(
        research, backend, target_model=selected["class"],
        evidence=RecLabToolbox(config.output_directory),
        max_steps=config.task.budget.max_model_turns,
        trace_path=config.output_directory / "traces" / f"agent-{uuid.uuid4().hex}.jsonl",
    )
