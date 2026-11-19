"""Research contracts plus opt-in integration with reviewed development tools.

The default diagnostic CLI remains read-only. OptimizationAgent requires an
explicit workspace, plan and trusted experiment runner; Worker formats stay intact.
"""

from .contracts import (
    EvidenceRef, ExperimentRecord, PlanReview, ResearchClaim,
    ResearchConclusion, ResearchPlan,
)
from .workflow import ResearchStage, require_plan_approval, stage_tool_names
from .toolbox import ResearchToolbox
from .optimization import OptimizationAgent

__all__ = [
    "EvidenceRef", "ExperimentRecord", "PlanReview", "ResearchClaim",
    "ResearchConclusion", "ResearchPlan", "ResearchStage",
    "require_plan_approval", "stage_tool_names",
    "ResearchToolbox", "OptimizationAgent",
]
