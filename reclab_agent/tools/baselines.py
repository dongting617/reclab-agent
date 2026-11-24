"""Baselines function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.tools.contracts import ToolError
from reclab_agent.tools.tool_group import ToolGroup

class BaselineTools(ToolGroup):
    definitions = (('resolve_baseline',
      'Resolve or establish the registered baseline in the background stage.',
      {}),)

    def resolve_baseline(self) -> dict[str, Any]:
        from reclab_agent.experiments.baselines import import_baseline
        from reclab_agent.experiments.outcomes import scored_record
        if self.active_stage != "background" or not hasattr(self, "baseline_context"):
            raise ToolError("WRONG_STAGE", "baseline resolution requires host-registered background context")
        config, inspection, output = self.baseline_context
        prior = next((v for v in self.experiments if v["candidate_id"] == self.plan.baseline_id), None)
        if prior:
            return scored_record(output, prior, self.plan)
        if config.baseline_report:
            return import_baseline(config.baseline_report, config, self.plan, inspection, self, output)
        record = self.run_experiment(self.plan.baseline_id, self.plan.seeds[0])
        return scored_record(output, record, self.plan)
