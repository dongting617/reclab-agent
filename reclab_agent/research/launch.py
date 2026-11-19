"""Launch one reviewed baseline/candidate optimization session from configuration."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from ..contracts import canonical_json, content_hash
from ..core import OpenAIResponsesBackend
from ..files import atomic_write, confined_path, read_regular_file
from ..tools import RecLabToolbox
from .binding import bind_task_tools
from .configuration import TaskConfiguration, load_task_configuration, read_json_strict
from .contracts import EvidenceRef, ExperimentRecord, PlanReview, ResearchPlan
from .inspection import TaskAdapterRegistry, inspect_task
from .launch_environment import build_remote_tools, load_agent_environment
from .optimization import OptimizationAgent


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
    if config.task.budget.max_training_jobs < 2 or config.task.budget.max_model_turns < 2:
        raise LaunchError("a baseline/candidate run requires at least two training jobs and two model turns")
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
        rejection_criteria="Reject failed/nonfinite candidates and improvements that violate a guardrail; no final-test or unmeasured quality claims.",
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
            tools=[{"type": "function", "name": "startup_check", "description": "Confirm tool-call compatibility.",
                    "parameters": {"type": "object", "properties": {}, "required": [],
                                   "additionalProperties": False}, "strict": True}],
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


def _scored_record(output, value, plan):
    record = ExperimentRecord.from_dict(value)
    result = {"record": record.to_dict()}
    if record.status != "completed":
        return result
    raw = read_regular_file(confined_path(output, record.score.source), max_bytes=512_000_000)
    if hashlib.sha256(raw).hexdigest() != record.score.sha256:
        raise LaunchError("independent scoring artifact hash mismatch")
    grade = json.loads(raw)
    metrics = grade["metrics"]
    if grade["protocol_hash"] != plan.protocol_hash or record.seed != plan.seeds[0]:
        raise LaunchError("independent score protocol or comparison seed mismatch")
    result["metrics"] = {name: metrics[name] for name in (plan.primary_metric, *plan.guardrail_metrics)}
    return result


def run_optimization(configuration, *, approve_plan=False, env_file=None, windows_ssh=False,
                     backend=None, registry=None, binder=bind_task_tools,
                     agent_factory=OptimizationAgent, progress=None):
    """An explicit CLI approval starts one fresh, bounded session.

    The CLI user approves the printed/configured scope with --approve-plan.
    The LLM cannot create its own review. Baseline and candidate use the same
    ResearchToolbox, so baseline consumes the same training/time budget.
    """
    if approve_plan is not True:
        raise LaunchError("inspect the plan command, then pass --approve-plan to authorize that configured scope")
    config = _configuration(configuration)
    if env_file is not None:
        load_agent_environment(env_file, config)
    registry = registry if registry is not None else TaskAdapterRegistry()
    effective, plan, inspection = prepare_launch(config, registry=registry)
    backend = _backend(effective, backend)
    if progress:
        progress("Checking Responses API and tool-call compatibility")
    api_check = _probe(backend)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = effective.output_directory / "sessions" / session_id
    # A fresh directory is a new session, never a disguised budget-recovery path.
    output.mkdir(parents=True, exist_ok=False)
    effective = replace(effective, output_directory=output)
    report = {"status": "running", "session_id": session_id, "session_directory": str(output),
              "report_path": str(output / "report.json"), "api_check": api_check,
              "plan_hash": plan.plan_hash, "baseline_id": effective.task.baseline_id,
              "primary_metric": plan.primary_metric, "comparison": [], "experiments": [],
              "test_evaluated": False, "cost_enforcement": "time/job limits only; provider and AutoDL billing not verified"}

    def write(name, value):
        atomic_write(confined_path(output, name), canonical_json(value).encode("utf-8"))

    review = PlanReview(plan.plan_hash, "approved", "cli-user",
        datetime.now(timezone.utc).isoformat(), "Explicit --approve-plan invocation for this configured scope.")
    write("plan.json", plan.to_dict())
    write("review.json", review.to_dict())
    write("report.json", report)
    research = None
    try:
        remote_tools = build_remote_tools(effective, windows_ssh=windows_ssh)
        research = binder(effective, plan=plan, review=review, registry=registry, remote_tools=remote_tools)
        if progress:
            progress("Session: " + str(output))
            progress("Training and independently scoring the unmodified baseline")
        baseline = research.run_experiment(effective.task.baseline_id, plan.seeds[0])
        baseline_score = _scored_record(output, baseline, plan)
        if "metrics" not in baseline_score:
            raise LaunchError("baseline experiment failed; candidate optimization was not started")
        report["experiments"] = [baseline_score]
        report["budget_usage"] = research.usage
        write("report.json", report)
        if progress:
            progress("Baseline metrics: " + canonical_json(baseline_score["metrics"]))
            progress("Starting the optimization Agent; trace: " + str(output / "agent.jsonl"))
        selected = read_json_strict(effective.model_config_path, max_bytes=1_000_000)["model"]
        agent = agent_factory(research, backend, target_model=selected["class"],
                              evidence=RecLabToolbox(output), max_steps=plan.budget.max_model_turns - 1,
                              trace_path=output / "agent.jsonl")
        prompt = (effective.task.question + "\nThe unmodified baseline has already run and used one training job. "
            "Inspect source and configuration, propose one scoped change, apply and validate it, then call "
            "run_experiment with a configured candidate_id (never the baseline_id) and seed " + str(plan.seeds[0]) +
            ". Read the independent score and report a measured positive, negative or inconclusive result in Chinese. "
            "Keep all actions within the supplied plan. Baseline evidence: " + canonical_json(baseline_score))
        report["agent_result"] = agent.run(prompt)
        scored = [_scored_record(output, value, plan) for value in research.experiments]
        report["experiments"] = scored
        for candidate in scored:
            record = candidate["record"]
            if record["candidate_id"] == effective.task.baseline_id or "metrics" not in candidate:
                continue
            deltas = {name: value - baseline_score["metrics"][name]
                      for name, value in candidate["metrics"].items()}
            guardrails = all(deltas[name] >= 0 for name in plan.guardrail_metrics)
            report["comparison"].append({"candidate_id": record["candidate_id"],
                "experiment_id": record["experiment_id"], "deltas": deltas,
                "passes_guardrails": guardrails,
                "improves": deltas[plan.primary_metric] > 0 and guardrails})
        report["status"] = "completed" if report["comparison"] else "incomplete"
        report["conclusion"] = ("Compare independent development scores; this is not final-test evidence."
                                 if report["comparison"] else "No independently scored candidate was produced.")
    except KeyboardInterrupt:
        report["status"] = "cancelled"
        raise
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error) if isinstance(error, LaunchError) else _public_error(error)
    finally:
        if research is not None:
            report["budget_usage"] = research.usage
            if report["status"] not in {"completed", "incomplete"}:
                report["experiments"] = [{"record": value} for value in research.experiments]
        write("report.json", report)
    return report
