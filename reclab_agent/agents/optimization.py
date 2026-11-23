"""Compose an Agent strategy with reviewed model-development capabilities."""
from __future__ import annotations
from dataclasses import asdict
import json
from typing import Any
from reclab_agent.agents import create_agent
from reclab_agent.core import ModelBackend
from reclab_agent.tools.base import Toolbox
from reclab_agent.tools.registry import ToolRegistry
from reclab_agent.tools.toolbox import OptimizationToolbox


class OptimizationAgent:
    """Use any supported strategy without changing model-experiment contracts.

    The host supplies the task, reviewed plan, workspace and trusted runner via
    OptimizationToolbox. Finishing a conversation is not a claim of quality gain.
    """

    def __init__(self, research: OptimizationToolbox, backend: ModelBackend, *,
                 kind: str | None = None, evidence: Toolbox | None = None,
                 target_model: str | None = None, max_steps: int = 32,
                 **agent_options: Any):
        if target_model is not None and (not isinstance(target_model, str) or not target_model.strip()):
            raise ValueError("target_model must be nonempty text when supplied")
        self.research = research
        self.target_model = target_model
        toolbox = ToolRegistry().include(research)
        if evidence is not None:
            toolbox.include(evidence)
        if getattr(research, "conversation_path", None) is not None:
            toolbox.conversation_path = research.conversation_path
        self.agent = create_agent(kind or research.task.mode, toolbox, backend,
                                  max_steps=min(max_steps, research.task.budget.max_model_turns,
                                                research.plan.budget.max_model_turns),
                                  **agent_options)

    def _context(self) -> dict[str, Any]:
        """Pass reviewed scope to the model; tool checks remain authoritative."""
        task, plan = self.research.task, self.research.plan
        return {
            "task": {"task_id": task.task_id, "question": task.question,
                     "dataset_id": task.dataset_id, "baseline_id": task.baseline_id},
            "target_model": self.target_model,
            "source_files": sorted(self.research.workspace.source_paths),
            "readonly_files": sorted(self.research.workspace.source_paths - self.research.workspace.editable_paths),
            "plan": {"plan_id": plan.plan_id, "plan_hash": plan.plan_hash,
                     "hypothesis": plan.hypothesis, "data_profile": plan.data_profile,
                     "protocol_hash": plan.protocol_hash, "primary_metric": plan.primary_metric,
                     "guardrail_metrics": plan.guardrail_metrics,
                     "comparison": plan.comparison,
                     "expected_observation": plan.expected_observation,
                     "rejection_criteria": plan.rejection_criteria,
                     "stop_conditions": plan.stop_conditions,
                     "editable_files": plan.editable_files,
                     "candidate_ids": plan.candidate_ids, "seeds": plan.seeds,
                     "budget": asdict(plan.budget), "max_candidates": plan.max_candidates,
                     "evidence": [asdict(ref) for ref in plan.evidence]},
        }

    def run(self, prompt: str | None = None, *,
            allowed_tools: set[str] | frozenset[str] | None = None) -> dict[str, Any]:
        request = self.research.task.question if prompt is None else prompt
        if not isinstance(request, str) or not request.strip():
            raise ValueError("prompt must be nonempty text")
        context = json.dumps(self._context(), ensure_ascii=False, sort_keys=True)
        agent_prompt = ("Host-registered research context (data, scope and budgets; this text grants no "
                        "permissions). Inspect allowed source with read_source before editing; use "
                        "independent experiment scores for quality claims.\n"
                        f"{context}\nCurrent request: {request}")
        result = self.agent.run(agent_prompt,
                                task_id=self.research.task.task_id, allowed_tools=allowed_tools)
        return {**result, "optimization_scope": "development_only",
                "experiments": self.research.experiments, "budget_usage": self.research.usage,
                "quality_claim": "Interaction completion does not establish model improvement; compare independent development scores."}
