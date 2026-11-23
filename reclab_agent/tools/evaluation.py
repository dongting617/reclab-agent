"""Evaluation function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.tools.contracts import ToolError
from reclab_agent.tools.group import ToolGroup

class EvaluationTools(ToolGroup):
    definitions = (('analyze_experiment',
      'Compare verified final and process metrics and produce a formal decision.',
      {'experiment_id': {'type': 'string'}}),)

    def analyze_experiment(self, experiment_id: str) -> dict[str, Any]:
        from reclab_agent.experiments.diagnostics import evaluate_process_metrics
        from reclab_agent.experiments.outcomes import scored_record
        from reclab_agent.experiments.outcomes import compare_experiments
        if self.active_stage != "analysis":
            raise ToolError("WRONG_STAGE", "formal decisions require the analysis stage")
        output = self.snapshots_root.parent
        baseline = next(v for v in self.experiments if v["candidate_id"] == self.plan.baseline_id)
        before = scored_record(output, baseline, self.plan)
        entry = scored_record(output, self.get_experiment(experiment_id), self.plan)
        if "metrics" not in entry:
            return {"entry": entry, "comparison": None}
        comparison = compare_experiments(before, entry, self.plan.primary_metric, self.plan.guardrail_metrics)
        definitions = (self.hypothesis or {}).get("process_metrics", [])
        process = evaluate_process_metrics(output, entry["record"], definitions)
        base_process = evaluate_process_metrics(output, baseline, definitions)
        required = any(x["purpose"] == "required" for x in definitions)
        comparable = bool(process["definition_hash"]) and process["definition_hash"] == base_process["definition_hash"]
        entry["process_metrics"] = process
        entry["eligible"] = process["passes_required"] and (not required or comparable)
        if not entry["eligible"]:
            comparison.update(improves=False, decision="inconclusive")
        else:
            comparison["decision"] = "retain" if comparison["improves"] else "reject"
        return {"entry": entry, "comparison": comparison}
