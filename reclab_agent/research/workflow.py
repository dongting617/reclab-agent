"""Stage/tool contract for the future coordinator, not a running state machine.

An Agent only exposes the tools registered for its task. Tool names
here specify intended phase permissions; this module does not register handlers,
launch jobs, debit budgets, persist reviews, or authenticate a reviewer.
"""
from enum import Enum
from types import MappingProxyType

from .contracts import PlanReview, ResearchPlan


class ResearchStage(str, Enum):
    CONTEXT = "context"
    INVESTIGATE = "investigate"
    REVIEW = "review"
    IMPLEMENT = "implement"
    EXPERIMENT = "experiment"
    ANALYZE = "analyze"
    CONFIRM = "confirm"
    REPORT = "report"


_READ = frozenset({
    "list_evidence", "inspect_artifact", "read_json_values", "compare_numeric_values",
    "diagnose_beam_gap", "diagnose_frequency_gap", "search_knowledge",
    "search_source", "read_source",
})
_RESULTS = frozenset({"get_experiment", "compare_experiments"})
_STAGE_TOOLS = MappingProxyType({
    ResearchStage.CONTEXT: _READ | _RESULTS,
    ResearchStage.INVESTIGATE: _READ | _RESULTS,
    ResearchStage.REVIEW: _READ | _RESULTS,
    ResearchStage.IMPLEMENT: _READ | {"apply_patch", "validate_candidate"},
    ResearchStage.EXPERIMENT: _READ | _RESULTS | {"submit_experiment", "cancel_experiment"},
    ResearchStage.ANALYZE: _READ | _RESULTS,
    ResearchStage.CONFIRM: _READ | _RESULTS | {"submit_experiment", "cancel_experiment"},
    ResearchStage.REPORT: _READ | _RESULTS,
})


def stage_tool_names(stage: ResearchStage | str) -> frozenset[str]:
    """Return the design allowlist; callers must intersect with real handlers."""
    return _STAGE_TOOLS[ResearchStage(stage)]


def require_plan_approval(plan: ResearchPlan, review: PlanReview | None) -> None:
    """Reject absent, rejected or stale reviews supplied by a trusted controller.

    This checks content binding only. The application must obtain the user's
    review through its own trusted channel before constructing PlanReview.
    """
    if (not isinstance(plan, ResearchPlan) or not isinstance(review, PlanReview)
            or review.decision != "approved" or review.plan_hash != plan.plan_hash):
        raise ValueError("the current research plan requires user review")
