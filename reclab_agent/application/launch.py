"""Launch one reviewed baseline/candidate optimization session from configuration."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.core import OpenAIResponsesBackend
from reclab_agent.infrastructure.files import (
    atomic_write,
    confined_path,
    read_regular_file,
    validated_directory,
)
from reclab_agent.execution.locking import job_lock
from reclab_agent.application.binding import bind_task_tools
from reclab_agent.domain.configuration import (
    TaskConfiguration,
    load_task_configuration,
    read_json_strict,
)
from reclab_agent.domain.research import EvidenceRef, PlanReview, ResearchPlan
from reclab_agent.experiments.adapters import TaskAdapterRegistry
from reclab_agent.application.inspection import inspect_task
from reclab_agent.application.environment import build_remote_tools, load_agent_environment
from reclab_agent.agents.optimization import OptimizationAgent
from reclab_agent.reporting import write_user_report
from reclab_agent.experiments.outcomes import scored_record as _scored_record
from reclab_agent.workflow.coordinator import CountedBackend, StudyCoordinator
from reclab_agent.memory.project import Project, knowledge_scope
from reclab_agent.workflow.resources import RunResources
from reclab_agent.application.history import archive_run, select_project_baseline
from reclab_agent.experiments.baselines import baseline_identity
from reclab_agent.integrations.tavily import LiteratureService
from reclab_agent.tools.connectivity import startup_check_schema


class LaunchError(ValueError):
    """Public launch failure whose message never includes provider credentials."""


def _configuration(value):
    return value if isinstance(value, TaskConfiguration) else load_task_configuration(value)


def prepare_launch(configuration, *, registry=None):
    """Describe a deterministic plan without API requests, jobs, or output writes."""
    config = _configuration(configuration)
    registry = registry if registry is not None else TaskAdapterRegistry()
    inspection = inspect_task(config, registry=registry)
    if not inspection["inputs_valid"]:
        checks = ", ".join(error["check"] for error in inspection["errors"])
        raise LaunchError(f"task input checks failed: {checks}")
    if config.task.baseline_id in config.candidate_ids:
        raise LaunchError("baseline_id must be separate from candidate_ids")
    effective = replace(config, candidate_ids=(config.task.baseline_id, *config.candidate_ids))
    model_config = read_regular_file(config.model_config_path, max_bytes=1_000_000)
    evidence = EvidenceRef(str(config.model_config_path), hashlib.sha256(model_config).hexdigest(),
                           "/", "Configured unmodified baseline; no candidate execution is implied.")
    plan = ResearchPlan(
        plan_id="launch-" + content_hash({"task": config.task.to_dict(),
            "source": inspection["source"]["tree_sha256"],
            "protocol": inspection["evaluation"]["protocol_hash"]})[:24],
        task_id=config.task.task_id, hypothesis=config.task.question,
        dataset_id=config.task.dataset_id, data_profile="Registered public train/valid data only; no test evaluation.",
        baseline_id=config.task.baseline_id, protocol_hash=inspection["evaluation"]["protocol_hash"],
        primary_metric=config.primary_metric, editable_files=config.editable_files,
        comparison="Train the unmodified configured baseline first; compare independently scored candidates at the same seed and resource limits.",
        expected_observation="An independently scored baseline and at least one candidate, with measured metric differences.",
        rejection_criteria="Reject failed/nonfinite candidates and improvements that violate a guardrail; use measured validation-set scores under one fixed training seed.",
        stop_conditions=("Stop when the configured model-turn, patch, training-job or time budget is exhausted.",
                         "Stop after failed baseline execution or unconfirmed remote stopping."),
        evidence=(evidence,), guardrail_metrics=config.guardrail_metrics,
        candidate_ids=effective.candidate_ids, seeds=(config.seeds[0],),
        budget=config.task.budget, max_candidates=len(effective.candidate_ids),
    )
    return effective, plan, inspection


def _backend(config, supplied=None):
    if supplied is not None:
        return supplied
    model, key = os.environ.get(config.model_env), os.environ.get(config.api_key_env)
    if not model or not key:
        raise LaunchError("configure the named Agent model and API-key environment variables")
    try:
        from openai import OpenAI
    except ImportError:
        raise LaunchError("install the optional SDK with: pip install -e '.[openai]'") from None
    # Bounded preflight and model requests. The Agent's own retry accounting is
    # retained; hidden SDK retries would make those counts misleading.
    return OpenAIResponsesBackend(model, client=OpenAI(api_key=key, timeout=60, max_retries=0))


def _public_error(error):
    # Provider exception bodies can echo input or credentials. Publish only the
    # class and optional HTTP status, not str(error), URLs, or response bodies.
    code = getattr(error, "status_code", None)
    return type(error).__name__ + (f" (HTTP {code})" if type(code) is int else "")


def _probe(backend):
    try:
        turn = backend.respond(
            instructions="Connectivity test only. Call startup_check with {}. Do not answer in text.",
            input_items=[{"role": "user", "content": "Call startup_check now."}],
            tools=[startup_check_schema()],
        )
        calls = [item for item in turn.output if item.get("type") == "function_call"]
        if not calls or calls[0].get("name") != "startup_check" or json.loads(calls[0].get("arguments", "")) != {}:
            raise LaunchError("the configured endpoint did not return the required Responses API tool call")
    except LaunchError:
        raise
    except Exception as error:
        raise LaunchError("Responses API/tool compatibility check failed: " + _public_error(error)) from None
    return {"responses_api": True, "tool_call": True, "gpu_started": False,
            "model_calls": 1, "usage": turn.usage}


def check_agent_api(configuration, *, env_file=None, backend=None):
    config = _configuration(configuration)
    if env_file is not None:
        load_agent_environment(env_file, config)
    result = _probe(_backend(config, backend))
    return {**result, "model": os.environ.get(config.model_env),
            "custom_base_url_configured": bool(os.environ.get("OPENAI_BASE_URL"))}


def run_optimization(configuration, *, approve_plan=False, env_file=None, windows_ssh=False,
                     backend=None, registry=None, binder=bind_task_tools,
                     agent_factory=OptimizationAgent, progress=None, resume_directory=None):
    """Run or resume one authorized four-stage study without resetting its limits."""
    if approve_plan is not True:
        raise LaunchError("inspect the plan command, then pass --approve-plan to authorize that configured scope")
    config = _configuration(configuration)
    if env_file is not None:
        load_agent_environment(env_file, config)
    registry = registry if registry is not None else TaskAdapterRegistry()
    effective, plan, inspection = prepare_launch(config, registry=registry)
    counted = CountedBackend(_backend(effective, backend), plan.budget.max_model_turns)
    resume = resume_directory is not None
    if resume:
        output = validated_directory(resume_directory)
        sessions = (config.output_directory / "sessions").absolute()
        if output.parent != sessions:
            raise LaunchError("resume directory must belong to this task's sessions")
        previous = read_json_strict(output / "report.json", max_bytes=32_000_000)
        if previous.get("registered_source"):
            config = replace(config, source_directory=Path(previous["registered_source"]))
            effective, plan, inspection = prepare_launch(replace(config, output_directory=output), registry=registry)
        if previous["plan_hash"] != plan.plan_hash or previous["status"] == "completed":
            raise LaunchError("resume requires an unfinished run with matching scope")
        if previous.get("baseline_identity") and previous["baseline_identity"] != baseline_identity(config, inspection):
            raise LaunchError("cannot resume under changed data, runtime or training limits")
        resources = RunResources(output, config, resume=True)
        counted.calls = resources.state["model_requests"]
        counted.before_request = resources.model_request
    else:
        if progress:
            progress("Checking Responses API and tool-call compatibility")
        # Keep the existing failed-preflight behavior: no output or GPU job.
        api_check = _probe(counted)
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        output = config.output_directory / "sessions" / session_id
        output.mkdir(parents=True, exist_ok=False)
        resources = RunResources(output, config)
        resources.state["model_requests"] = counted.calls
        resources.save()
        counted.before_request = resources.model_request
    with job_lock(output, "research-session") as acquired:
        if not acquired:
            raise LaunchError("another controller owns this research session")
        project = Project(config.project_directory or config.output_directory / "project", retrieval=config.retrieval)
        resources.project = project
        counted.deadline = resources.state["deadline"]
        scope = knowledge_scope(config, inspection)
        project.configure(config, scope, inspection)
        if not resume:
            config = select_project_baseline(project, scope, config, inspection, registry=registry, output_directory=output)
            effective, plan, inspection = prepare_launch(replace(config, output_directory=output), registry=registry)
        effective = replace(effective, output_directory=output, resume_jobs=resume)
        selected = read_json_strict(effective.model_config_path, max_bytes=1_000_000)["model"]
        report_context = {"task_id": effective.task.task_id, "task_question": effective.task.question,
            "target_model": selected["class"], "agent_mode": effective.task.mode,
            "dataset_id": effective.task.dataset_id, "seed": plan.seeds[0],
            "primary_metric": plan.primary_metric, "guardrail_metrics": list(plan.guardrail_metrics),
            "editable_files": list(plan.editable_files)}
        report = previous if resume else {"status": "running", "session_id": output.name,
            "session_directory": str(output), "report_path": str(output / "report.json"),
            "api_check": api_check, "plan_hash": plan.plan_hash, "baseline_id": effective.task.baseline_id,
            "primary_metric": plan.primary_metric, "comparison": [], "experiments": [], "context": report_context,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "test_evaluated": False,
            "cost_enforcement": "time/request/job limits only; billing not verified"}
        if resume and "error" in report:
            report.setdefault("previous_failures", []).append({
                name: report.get(name) for name in ("status", "error", "finished_at_utc")})
            report.pop("error")
        report.update(status="running", project_directory=str(project.root), knowledge_scope=scope,
                      registered_source=str(effective.source_directory))
        def write(name, value):
            atomic_write(confined_path(output, name), canonical_json(value).encode("utf-8"))
        review = PlanReview(plan.plan_hash, "approved", "cli-user",
            datetime.now(timezone.utc).isoformat(), "Explicit start/resume authorization within this scope.")
        write("plan.json", plan.to_dict())
        write("review.json", review.to_dict())
        write("report.json", report)
        research = None
        started = time.monotonic()
        try:
            resources.check()
            remote_tools = build_remote_tools(effective, windows_ssh=windows_ssh)
            research = binder(effective, plan=plan, review=review, registry=registry, remote_tools=remote_tools)
            research.context.attach(project, scope, output, resources.state["deadline"], resume=resume)
            research.attach_knowledge(LiteratureService(project, scope, resources.network_request,
                                                        key_env=config.tavily_key_env))
            StudyCoordinator(effective, plan, inspection, research, counted, agent_factory,
                                report, write, progress=progress).run()
        except KeyboardInterrupt:
            report["status"] = "cancelled"
            raise
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error) if isinstance(error, LaunchError) else _public_error(error)
        finally:
            if research is not None:
                report["budget_usage"] = {**research.usage, **resources.state}
                if report["status"] not in {"completed", "incomplete"}:
                    verified = {x["record"]["experiment_id"]: x for x in report["experiments"] if "metrics" in x}
                    report["experiments"] = [verified.get(v["experiment_id"], {"record": v}) for v in research.experiments]
                research.save_state()
                try:
                    archive_run(project, scope, effective, plan, inspection, research, report)
                except Exception as error:
                    report["archive_error"] = _public_error(error)
            report["elapsed_seconds"] = report.get("elapsed_seconds", 0) + time.monotonic() - started
            report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            from reclab_agent.evaluation.costs import summarize_usage
            usage_events = [{"event": "model_turn", "usage": report.get("api_check", {}).get("usage")}]
            trace = output / "agent.jsonl"
            if trace.exists():
                try:
                    usage_events.extend(json.loads(line) for line in read_regular_file(trace, max_bytes=64_000_000).decode("utf-8").splitlines() if line.strip())
                except (OSError, ValueError):
                    usage_events.append({"event": "model_error"})
            report["resource_usage"] = summarize_usage(usage_events)
            report["requires_reconciliation"] = any(confined_path(output, name).exists()
                for name in ("active-remote-job.json", "execution-blocked.json"))
            write("report.json", report)
            try:
                report.update(write_user_report(output, report, context=report_context))
            except Exception as error:
                report["user_report_error"] = _public_error(error)
            write("report.json", report)
            if report["status"] == "cancelled" and progress:
                progress("User report: " + report.get("user_report_path", report["report_path"]))
        return report
