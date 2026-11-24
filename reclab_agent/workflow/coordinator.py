"""Four-stage research orchestration over registered function tools."""
from __future__ import annotations
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.tools import RecLabToolbox
from reclab_agent.experiments.baselines import baseline_identity, checkpoint_manifest
from reclab_agent.workflow.stages import StudyStage


class RequestLimitExceeded(RuntimeError):
    request_not_sent = True


class CountedBackend:
    """One request allowance shared by probes, phases and SDK-level retries."""
    def __init__(self, backend, limit):
        self.backend, self.limit, self.calls = backend, limit, 0
        self.model = getattr(backend, "model", None)
        self.before_request = None
        self.deadline = None

    def respond(self, **kwargs):
        if self.calls >= self.limit:
            raise RequestLimitExceeded("shared model request budget exhausted")
        self.calls += 1
        if self.before_request:
            self.before_request(self.calls)
        if self.deadline is not None:
            import time
            from reclab_agent.core import OpenAIResponsesBackend
            if isinstance(self.backend, OpenAIResponsesBackend):
                self.backend.client = self.backend.client.with_options(timeout=max(.1, min(60, self.deadline - time.time())))
        return self.backend.respond(**kwargs)


class StudyCoordinator:
    def __init__(self, config, plan, inspection, research, backend, agent_factory,
                 report, write, *, progress=None):
        self.config, self.plan, self.inspection = config, plan, inspection
        self.research, self.backend, self.agent_factory = research, backend, agent_factory
        self.report, self.write, self.progress = report, write, progress
        self.output = config.output_directory

    def stage(self, stage, **payload):
        self.research.active_stage = stage.value
        self.research.active_candidate_id = payload.get("candidate_id")
        self.report.setdefault("stages", []).append({"stage": stage.value, **payload})
        self.report["budget_usage"] = {**self.research.usage, "model_requests": self.backend.calls}
        self.research.save_state()
        self.write("report.json", self.report)
        if self.progress:
            self.progress("Research stage: " + stage.value)

    def invoke(self, name, **arguments):
        result = self.research.invoke(name, arguments)
        if not result.get("ok"):
            self.report.setdefault("tool_errors", []).append({"tool": name, **result["error"]})
            self.write("report.json", self.report)
            raise ValueError(result["error"]["code"] + ": " + result["error"]["message"])
        return result["result"]

    def agent(self, prompt, allowed_tools, *, kind=None):
        phase = self.research.active_stage
        candidate = self.research.active_candidate_id or "background"
        self.research.conversation_path = self.output / "conversations" / f"{phase}-{candidate}.json"
        options = {"target_model": self.report["context"]["target_model"],
                   "evidence": RecLabToolbox(self.output),
                   "max_steps": max(1, self.backend.limit - self.backend.calls),
                   "trace_path": self.output / "agent.jsonl"}
        if kind:
            options["kind"] = kind
        return self.agent_factory(self.research, self.backend, **options).run(prompt, allowed_tools=allowed_tools)

    def run(self):
        self.stage(StudyStage.BACKGROUND)
        initial = self.research.workspace.snapshot(self.research.snapshots_root)
        self.research.baseline_source_hash = initial["source_hash"]
        self.report["background"] = {"task": self.config.task.question, "source_hash": initial["source_hash"],
                                     "dataset_id": self.plan.dataset_id, "missing": []}
        self.report["baseline_identity"] = baseline_identity(self.config, self.inspection)
        if self.research.project:
            self.report["background"]["related_experience"] = self.research.search_knowledge(self.config.task.question)["matches"]
        reference = self.config.baseline_report
        self.research.context.baseline_context = (self.config, self.inspection, self.output)
        baseline = self.invoke("resolve_baseline")
        old_baseline = next((x for x in self.report.get("experiments", [])
                             if x["record"]["experiment_id"] == baseline["record"]["experiment_id"]), None)
        if old_baseline and old_baseline.get("reused_from"):
            baseline["reused_from"] = old_baseline["reused_from"]
        if "metrics" not in baseline:
            raise ValueError("baseline experiment failed; candidate optimization was not started")
        previous_entries = self.report.get("experiments", [])
        self.report["experiments"] = [baseline, *[x for x in previous_entries if x["record"]["experiment_id"] != baseline["record"]["experiment_id"]]]
        self.report["baseline_reused"] = bool(reference) or bool(baseline.get("reused_from"))
        self.report["baseline_artifacts"] = checkpoint_manifest(self.output, baseline)
        best = baseline
        for entry in self.report["experiments"][1:]:
            if entry.get("eligible", True) and entry.get("metrics", {}).get(self.plan.primary_metric, -1) > best["metrics"][self.plan.primary_metric]:
                best = entry
        for candidate_id in self.config.candidate_ids:
            if candidate_id == self.plan.baseline_id:
                continue
            usage = self.research.usage
            recorded = any(v["candidate_id"] == candidate_id for v in self.research.experiments)
            pending = self.research.pending_request is not None
            if not recorded and not pending and (usage["experiment_attempts"] >= self.plan.budget.max_training_jobs
                    or self.backend.calls >= self.backend.limit):
                break
            if any(x["candidate_id"] == candidate_id for x in self.report["comparison"]):
                continue
            self.stage(StudyStage.HYPOTHESIS, candidate_id=candidate_id)
            saved_hypothesis = next((h for h in self.report.get("hypotheses", []) if h.get("candidate_id") == candidate_id), None)
            self.research.hypothesis = saved_hypothesis
            context = {"baseline": baseline, "current_best": best,
                       "previous_decisions": self.report.get("comparison", []),
                       "candidate_id": candidate_id, "background": self.report["background"]}
            if not saved_hypothesis:
                self.agent("Register one testable hypothesis with record_hypothesis. Include expected causal chain, "
                       "risks, evidence gaps and diagnostic/required observations. Do not edit or train yet.\n"
                       + canonical_json(context), {"record_hypothesis", "search_knowledge", "list_experiments",
                           "search_literature", "fetch_literature", "read_source"} & {s.name for s in self.research.specs}, kind="react")
            if self.research.hypothesis is None:
                break
            if not saved_hypothesis:
                self.report.setdefault("hypotheses", []).append({**self.research.hypothesis, "candidate_id": candidate_id})
            self.research.active_candidate_id = candidate_id
            self.stage(StudyStage.EXPERIMENT, candidate_id=candidate_id)
            terminal = any(v["candidate_id"] == candidate_id for v in self.research.experiments)
            if self.research.pending_request:
                request = self.research.pending_request["request"]
                self.invoke("run_experiment", candidate_id=request[1], seed=request[2])
                terminal = True
            if not terminal:
                self.report["agent_result"] = self.agent(
                self.config.task.question + "\nExecute the registered hypothesis. Inspect source, edit and validate; "
                f"run candidate_id={candidate_id}, seed={self.plan.seeds[0]}. Read logs and independent scores. "
                "Do not use another candidate_id. Registered context:\n" + canonical_json({**context,
                    "hypothesis": self.research.hypothesis}),
                {s.name for s in self.research.specs} - {"record_hypothesis"})
                if self.config.task.mode == "research_audit":
                    self.report.setdefault("audits", {})[candidate_id] = {
                        key: self.report["agent_result"].get(key) for key in
                        ("status", "audit", "audit_approved", "evidence_packet_hash")}
            self.stage(StudyStage.ANALYSIS, candidate_id=candidate_id)
            for value in self.research.experiments:
                if value["candidate_id"] != candidate_id:
                    continue
                analyzed = self.invoke("analyze_experiment", experiment_id=value["experiment_id"])
                entry, comparison = analyzed["entry"], analyzed["comparison"]
                if self.config.task.mode == "research_audit":
                    audit = self.report.get("audits", {}).get(candidate_id, {})
                    entry["audit"] = audit
                    if audit.get("status") != "completed" or audit.get("audit_approved") is not True:
                        entry["eligible"] = False
                        if comparison is not None:
                            comparison.update(improves=False, decision="inconclusive", audit_required=True)
                if not any(x["record"]["experiment_id"] == value["experiment_id"] for x in self.report["experiments"]):
                    self.report["experiments"].append(entry)
                if comparison is None:
                    continue
                self.report["comparison"].append(comparison)
                if comparison["improves"] and entry["metrics"][self.plan.primary_metric] > best["metrics"][self.plan.primary_metric]:
                    best = entry
            self.research.restore_source(best["record"]["source_hash"])
        if self.research.active_stage != StudyStage.ANALYSIS.value:
            self.stage(StudyStage.ANALYSIS)
        self.report["best_experiment_id"] = best["record"]["experiment_id"]
        self.report["status"] = "completed" if self.report["comparison"] else "incomplete"
        self.report["conclusion"] = "Compare independent validation-set scores under the same fixed training seed."
        return self.report
