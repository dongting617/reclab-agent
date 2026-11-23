"""Task-scope authorization, independent of research stages."""
from reclab_agent.domain.research import PlanReview, ResearchPlan


def require_plan_approval(plan: ResearchPlan, review: PlanReview | None) -> None:
    """Reject absent, rejected or stale reviews supplied by a trusted controller.

    This checks content binding only. The application must obtain the user's
    review through its own trusted channel before constructing PlanReview.
    """
    if (not isinstance(plan, ResearchPlan) or not isinstance(review, PlanReview)
            or review.decision != "approved" or review.plan_hash != plan.plan_hash):
        raise ValueError("the current research plan requires user review")
