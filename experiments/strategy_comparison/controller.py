"""Compare independent Agent strategies with equal, cumulative search budgets."""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
import difflib
import hashlib
from pathlib import Path
import time
import uuid
from reclab_agent.execution.locking import job_lock
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.execution.worker import InputTree
from reclab_agent.infrastructure.files import atomic_write, confined_path, read_regular_file
from reclab_agent.tools import RecLabToolbox
from reclab_agent.application.binding import bind_task_tools
from .tools import CountedBackend, RoundResearch, restore_snapshot
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.research import EvidenceRef, PlanReview
from reclab_agent.experiments.adapters import TaskAdapterRegistry
from reclab_agent.application.launch import (
    LaunchError,
    _backend,
    _configuration,
    _probe,
    _public_error,
    _scored_record,
    prepare_launch,
)
from reclab_agent.application.environment import build_remote_tools, load_agent_environment
from reclab_agent.agents.optimization import OptimizationAgent
from reclab_agent.reporting import write_user_report


def _write(root, name, value):
    atomic_write(confined_path(root, name), canonical_json(value).encode("utf-8"))


def _unconfirmed(root):
    return any(confined_path(root, name).exists()
               for name in ("active-remote-job.json", "execution-blocked.json"))


def _best_entry(entry, source_directory, score_root):
    return {**entry, "source_directory": str(source_directory), "score_root": str(score_root)}


def _changes(research, before_hash, after_hash):
    changes = []
    for name in sorted(research.workspace.editable_paths):
        before = read_regular_file(confined_path(research.snapshots_root, before_hash + "/" + name),
                                   max_bytes=1_000_000).decode("utf-8").replace("\r\n", "\n")
        after = read_regular_file(confined_path(research.snapshots_root, after_hash + "/" + name),
                                  max_bytes=1_000_000).decode("utf-8").replace("\r\n", "\n")
        if before != after:
            diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile=name, tofile=name))
            changes.append({"path": name, "diff": diff[:12_000], "truncated": len(diff) > 12_000})
    return changes


def _run_strategy(config, global_plan, baseline, baseline_bytes, strategy, backend,
                  registry, binder, agent_factory, output, state, progress,
                  *, global_output, global_report, windows_ssh):
    comparison = config.comparison
    budget = comparison.strategy_budget(config.task.budget)
    candidates = config.candidate_ids[:comparison.candidates_per_strategy]
    task_mode = {"react": "single", "reflection": "reviewer"}.get(strategy, strategy)
    child = replace(config, output_directory=output,
                    task=replace(config.task, mode=task_mode, budget=budget), candidate_ids=candidates)
    baseline_ref = EvidenceRef("shared-baseline/grade.json", baseline["record"]["score"]["sha256"],
                              "/metrics/" + global_plan.primary_metric,
                              "Shared baseline scored once under the global comparison plan; not a new strategy job.")
    plan = replace(global_plan, plan_id=global_plan.plan_id + "-" + strategy,
                   hypothesis=f"{strategy}: {config.task.question}", budget=budget,
                   candidate_ids=candidates, max_candidates=len(candidates),
                   evidence=(*global_plan.evidence, baseline_ref))
    review = PlanReview(plan.plan_hash, "approved", "comparison-controller",
                        datetime.now(timezone.utc).isoformat(),
                        "Derived from explicit comparison startup authorization; identical allocated budget. "
                        "Global plan: " + global_plan.plan_hash)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(confined_path(output, baseline_ref.source), baseline_bytes)
    _write(output, "plan.json", plan.to_dict())
    _write(output, "review.json", review.to_dict())
    tools = build_remote_tools(child, windows_ssh=windows_ssh)
    research = binder(child, plan=plan, review=review, registry=registry, remote_tools=tools)
    initial = research.workspace.snapshot(research.snapshots_root)
    if initial["source_hash"] != baseline["record"]["source_hash"]:
        raise LaunchError("strategy source differs from the shared baseline")
    best = _best_entry(baseline, initial["directory"], global_output / "baseline")
    counted = CountedBackend(backend, budget.max_model_turns)
    seen = {initial["source_hash"]}
    state.update(status="running", best=best, budget=asdict(budget), rounds=[],
                 candidate_count=0, model_calls=0, model_usage=counted.usage)
    selected_model = read_json_strict(child.model_config_path, max_bytes=1_000_000)["model"]["class"]
    started = time.monotonic()
    try:
        for index, candidate_id in enumerate(candidates, 1):
            remaining = budget.max_model_turns - counted.calls
            if remaining <= 0 or research.usage["experiment_attempts"] >= budget.max_training_jobs:
                state["stop_reason"] = "strategy budget exhausted"
                break
            if _unconfirmed(output):
                raise LaunchError("strategy has an unconfirmed remote job; comparison stopped")
            starting = restore_snapshot(research, best["record"]["source_hash"])
            proxy = RoundResearch(research, candidate_id, plan.seeds[0], seen)
            before = len(research.experiments)
            model_before = counted.calls
            round_state = {"round": index, "candidate_id": candidate_id,
                           "starting_source_hash": starting["source_hash"], "experiments": []}
            state["rounds"].append(round_state)
            if progress:
                progress(f"策略 {strategy}：候选 {index}/{len(candidates)}")
            options = ({"max_plan_steps": comparison.max_plan_steps} if strategy == "plan_execute" else
                       {"max_reflections": comparison.max_reflections} if strategy == "reflection" else {})
            # Equal total model-call ceilings, with unused calls carried within
            # this strategy only. Planner/critic calls and retries count too.
            per_round_turns = max(1, remaining // (len(candidates) - index + 1))
            agent = agent_factory(proxy, counted, kind=strategy, target_model=selected_model,
                                  evidence=RecLabToolbox(output), max_steps=per_round_turns,
                                  trace_path=output / "traces" / f"round-{index}.jsonl", **options)
            history = [{"round": item["round"], "candidate_id": item["candidate_id"],
                        "experiments": [{"record": {key: entry["record"][key] for key in
                            ("experiment_id", "candidate_id", "source_hash", "status", "score")},
                            "metrics": entry.get("metrics")} for entry in item["experiments"]],
                        "changes": item.get("source_changes", []),
                        "decision": item.get("decision"), "answer": item.get("agent_answer", "")[:2000]}
                       for item in state["rounds"][:-1]]
            prompt = (config.task.question + "\nThis is an independent strategy comparison. "
                f"Your assigned strategy is {strategy}. Improve {plan.primary_metric} while satisfying "
                "the guardrail metrics against the shared baseline; do not run or evaluate other strategies yourself. "
                "Only your own previous rounds are shown. The shared baseline is already trained. "
                "Start from the current independently verified best source. Propose a new scoped hypothesis, "
                "read source, edit and validate, and run exactly one candidate experiment in this round. "
                f"Use candidate_id={candidate_id}, seed={plan.seeds[0]}. Do not rerun an already tested source. "
                "After its independent score, analyze results without starting another training in this round. "
                "Write the hypothesis, observations and next useful step in Chinese. "
                "Training starts from scratch; source inheritance is not checkpoint warm-start.\n" + canonical_json({
                    "shared_baseline": {"metrics": baseline["metrics"], "score": baseline_ref.to_dict()},
                    "current_best": {"source_hash": best["record"]["source_hash"], "metrics": best["metrics"]},
                    "own_history": history, "strategy_budget_remaining": {
                        "model_calls": remaining,
                        "training_attempts": budget.max_training_jobs - research.usage["experiment_attempts"],
                        "patches": budget.max_patches - research.usage["patch_attempts"]}}))
            try:
                result = agent.run(prompt)
                round_state.update(agent_status=result.get("status"), agent_answer=result.get("answer", ""))
            except Exception as error:
                round_state.update(agent_status="failed", agent_error=_public_error(error))
            if _unconfirmed(output):
                raise LaunchError("strategy has an unconfirmed remote job; comparison stopped")
            new_entries = [_scored_record(output, value, plan) for value in research.experiments[before:]]
            round_state["experiments"] = new_entries
            round_state["source_changes"] = [change for entry in new_entries for change in
                _changes(research, starting["source_hash"], entry["record"]["source_hash"])]
            decision = "no_experiment"
            for entry in new_entries:
                seen.add(entry["record"]["source_hash"])
                decision = "failed" if "metrics" not in entry else "rejected"
                if "metrics" not in entry:
                    continue
                state["candidate_count"] += 1
                guardrails = all(entry["metrics"][name] >= baseline["metrics"][name]
                                 for name in plan.guardrail_metrics)
                if guardrails and entry["metrics"][plan.primary_metric] > best["metrics"][plan.primary_metric]:
                    snapshot = confined_path(research.snapshots_root, entry["record"]["source_hash"])
                    best = _best_entry(entry, snapshot, output)
                    decision = "accepted"
            restore_snapshot(research, best["record"]["source_hash"])
            round_state.update(decision=decision, model_calls=counted.calls - model_before,
                               best_after={"source_hash": best["record"]["source_hash"], "metrics": best["metrics"]})
            state.update(best=best, budget_usage=research.usage, model_calls=counted.calls, model_usage=counted.usage)
            _write(output, "report.json", state)
            _write(global_output, "report.json", global_report)
        state["status"] = "completed" if state["candidate_count"] == len(candidates) else "incomplete"
    finally:
        # Interrupted/scoring-failed rounds still retain their actual attempts.
        # These raw records carry no accepted metrics or promotion decision.
        for item in state["rounds"]:
            if not item["experiments"]:
                item["experiments"] = [{"record": value} for value in research.experiments
                    if value["candidate_id"] == item["candidate_id"]]
        state.update(best=best, budget_usage=research.usage, model_calls=counted.calls,
                     model_usage=counted.usage, elapsed_seconds=time.monotonic() - started)
        _write(output, "report.json", state)
    return state


def run_comparison(configuration, *, approve_plan=False, env_file=None, windows_ssh=False,
                   backend=None, registry=None, binder=bind_task_tools,
                   agent_factory=OptimizationAgent, progress=None):
    """One shared baseline and isolated, equally bounded adaptive searches."""
    if approve_plan is not True:
        raise LaunchError("explicit startup approval is required for the configured comparison")
    config = _configuration(configuration)
    if config.comparison is None:
        raise LaunchError("configure a strategy comparison before starting this entrypoint")
    config.comparison.validate_budget(config.task.budget)
    if len(config.candidate_ids) < config.comparison.candidates_per_strategy:
        raise LaunchError("configure enough candidate IDs for every strategy")
    if env_file is not None:
        load_agent_environment(env_file, config)
    registry = registry if registry is not None else TaskAdapterRegistry()
    effective, plan, inspection = prepare_launch(config, registry=registry)
    backend = _backend(effective, backend)
    base_output = config.output_directory
    with job_lock(base_output, "strategy-comparison") as acquired:
        if not acquired:
            raise LaunchError("another comparison is already running")
        active = confined_path(base_output, "comparison-active.json")
        if active.exists():
            raise LaunchError("a prior comparison needs reconciliation; inspect comparison-active.json before restarting")
        if progress:
            progress("Checking Responses API and tool-call compatibility")
        api_check = _probe(backend)
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-compare-" + uuid.uuid4().hex[:8]
        output = base_output / "sessions" / session_id
        output.mkdir(parents=True, exist_ok=False)
        baseline_output = output / "baseline"
        report = {"kind": "strategy_comparison", "status": "running", "session_id": session_id,
                  "session_directory": str(output), "report_path": str(output / "report.json"),
                  "api_check": api_check, "plan_hash": plan.plan_hash, "seed": plan.seeds[0],
                  "primary_metric": plan.primary_metric, "strategies": [], "comparison": [],
                  "winning_strategies": [], "global_budget": asdict(config.task.budget),
                  "candidate_limit_per_strategy": config.comparison.candidates_per_strategy,
                  "test_evaluated": False, "source_inheritance": "within-strategy only; every training starts from scratch",
                  "limitations": ["One training seed and one search per strategy; no statistical superiority claim.",
                                  "Only independent validation-set scores select best sources; one fixed training seed per strategy.",
                                  "Provider and AutoDL billing are not automatically enforced."]}
        report_context = {
            "task_question": effective.task.question, "task_id": effective.task.task_id,
            "target_model": read_json_strict(effective.model_config_path, max_bytes=1_000_000)["model"]["class"],
            "agent_mode": "strategy_comparison", "dataset_id": effective.task.dataset_id,
            "seed": plan.seeds[0], "primary_metric": plan.primary_metric,
            "guardrail_metrics": list(plan.guardrail_metrics), "editable_files": list(plan.editable_files),
        }
        report["context"] = report_context
        review = PlanReview(plan.plan_hash, "approved", "cli-user", datetime.now(timezone.utc).isoformat(),
                            "Explicit startup authorization for the configured independent strategy comparison.")
        _write(output, "plan.json", plan.to_dict())
        _write(output, "review.json", review.to_dict())
        _write(base_output, active.name, {"session_directory": str(output), "plan_hash": plan.plan_hash})
        _write(output, "report.json", report)
        baseline_research = None
        child_outputs = []
        started = time.monotonic()
        try:
            baseline_config = replace(effective, output_directory=baseline_output)
            tools = build_remote_tools(baseline_config, windows_ssh=windows_ssh)
            baseline_research = binder(baseline_config, plan=plan, review=review, registry=registry, remote_tools=tools)
            if progress:
                progress("Training and independently scoring the unmodified baseline")
            baseline_record = baseline_research.run_experiment(config.task.baseline_id, plan.seeds[0])
            if _unconfirmed(baseline_output):
                raise LaunchError("baseline stopping is unconfirmed; comparison stopped")
            baseline = _scored_record(baseline_output, baseline_record, plan)
            if "metrics" not in baseline:
                raise LaunchError("shared baseline failed; strategy searches were not started")
            report["baseline"] = baseline
            baseline_bytes = read_regular_file(confined_path(baseline_output, baseline_record["score"]["source"]),
                                               max_bytes=512_000_000)
            if hashlib.sha256(baseline_bytes).hexdigest() != baseline_record["score"]["sha256"]:
                raise LaunchError("shared baseline scoring artifact changed")
            original = baseline_research.workspace.snapshot(baseline_research.snapshots_root)
            report["selected_source_directory"] = original["directory"]
            if progress:
                progress("Baseline metrics: " + canonical_json(baseline["metrics"]))
            for strategy in config.comparison.strategies:
                if InputTree.freeze(config.source_directory, "source").sha256 != inspection["source"]["tree_sha256"]:
                    raise LaunchError("registered original source changed during the comparison")
                child_output = output / "strategies" / strategy
                child_outputs.append(child_output)
                state = {"strategy": strategy, "status": "running"}
                report["strategies"].append(state)
                try:
                    _run_strategy(config, plan, baseline, baseline_bytes, strategy, backend, registry,
                                  binder, agent_factory, child_output, state, progress,
                                  global_output=output, global_report=report, windows_ssh=windows_ssh)
                finally:
                    _write(output, "report.json", report)
            primary = plan.primary_metric
            for state in report["strategies"]:
                best = state["best"]
                deltas = {name: value - baseline["metrics"][name] for name, value in best["metrics"].items()}
                entries = [entry for item in state["rounds"] for entry in item["experiments"] if "metrics" in entry]
                best_candidate = max(entries, key=lambda entry: entry["metrics"][primary], default=None)
                server_times = []
                for item in state["rounds"]:
                    for entry in item["experiments"]:
                        execution_path = output / "strategies" / state["strategy"] / "experiments" / entry["record"]["experiment_id"] / "execution.json"
                        if execution_path.exists():
                            execution = read_json_strict(execution_path)
                            if "server_job_seconds" in execution:
                                server_times.append(execution["server_job_seconds"])
                report["comparison"].append({"strategy": state["strategy"], "best_metrics": best["metrics"],
                    "best_candidate": best_candidate, "deltas": deltas,
                    "improves": deltas[primary] > 0, "candidate_count": state["candidate_count"],
                    "training_attempts": state["budget_usage"]["experiment_attempts"],
                    "model_calls": state["model_calls"], "model_usage": state["model_usage"],
                    "elapsed_seconds": state["elapsed_seconds"],
                    "training_prediction_seconds": sum(server_times) if server_times else None})
            complete = all(state["status"] == "completed" for state in report["strategies"])
            report["status"] = "completed" if complete else "incomplete"
            if complete:
                highest = max(item["best_metrics"][primary] for item in report["comparison"])
                report["winning_strategies"] = [item["strategy"] for item in report["comparison"]
                    if item["improves"] and abs(item["best_metrics"][primary] - highest) <= 1e-12]
                if report["winning_strategies"]:
                    winner = next(state for state in report["strategies"]
                                  if state["strategy"] == report["winning_strategies"][0])
                    report["selected_source_directory"] = winner["best"]["source_directory"]
            report["conclusion"] = ("Preliminary comparison from equal budgets; inspect independent scores and resource use."
                                     if complete else "Not every strategy produced the requested scored candidates; no winner declared.")
        except KeyboardInterrupt:
            report["status"] = "cancelled"
            raise
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error) if isinstance(error, LaunchError) else _public_error(error)
        finally:
            usages = [state.get("budget_usage", {}) for state in report["strategies"]]
            baseline_usage = baseline_research.usage if baseline_research is not None else {}
            if "baseline" not in report and baseline_research is not None and baseline_research.experiments:
                report["baseline"] = {"record": baseline_research.experiments[0]}
            report["budget_usage"] = {
                "experiment_attempts": baseline_usage.get("experiment_attempts", 0) + sum(u.get("experiment_attempts", 0) for u in usages),
                "patch_attempts": sum(u.get("patch_attempts", 0) for u in usages),
                "reserved_training_seconds": baseline_usage.get("reserved_training_seconds", 0) + sum(u.get("reserved_training_seconds", 0) for u in usages),
                "model_calls": 1 + sum(state.get("model_calls", 0) for state in report["strategies"]),
            }
            report["elapsed_seconds"] = time.monotonic() - started
            unsafe = _unconfirmed(baseline_output) or any(_unconfirmed(path) for path in child_outputs)
            report["requires_reconciliation"] = unsafe
            try:
                report.update(write_user_report(output, report, context=report_context))
            except Exception as error:
                report["user_report_error"] = _public_error(error)
            _write(output, "report.json", report)
            if not unsafe:
                active.unlink(missing_ok=True)
        return report
