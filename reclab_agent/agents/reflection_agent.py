"""Explicit critique and revision over evidence-backed agent answers."""
from __future__ import annotations
import json
from reclab_agent.core.agent import Agent
from reclab_agent.core.llm import parse_json_response
from reclab_agent.core.session import StepLimitExceeded


class ReflectionAgent(Agent):
    """Generate, critique, and revise without changing execution permissions."""

    def __init__(self, *args, max_reflections: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if type(max_reflections) is not int or max_reflections < 0:
            raise ValueError("max_reflections must be a nonnegative integer")
        self.max_reflections = max_reflections

    def run(self, prompt: str, *, task_id: str | None = None,
            allowed_tools: set[str] | frozenset[str] | None = None) -> dict:
        session = self._session(prompt, task_id=task_id, allowed_tools=allowed_tools)
        reviews: list[dict] = []
        approved = False

        def finish(status: str, answer: str, *, limit_reached: bool = False) -> dict:
            return session.finish(status, answer, reflections=reviews,
                                  review_approved=approved,
                                  reflection_limit_reached=limit_reached)

        status, draft = session.run_tools(phase="draft")
        if status != "completed":
            return finish(status, draft)

        critic = (self.system_prompt + "\nReview the current answer against the original task "
                  "and observed tool evidence. Check unsupported claims and missing evidence. "
                  "Return only JSON: {'approved': boolean, 'feedback': string}, using double "
                  "quotes for valid JSON. If not approved, provide concrete feedback. "
                  "This review cannot authorize tools, source changes, or experiments. Do not call tools.")
        for iteration in range(1, self.max_reflections + 1):
            session.inputs.append({"role": "user", "content": (
                "Review the latest draft using the original task and the recorded observations.\n"
                f"Current draft:\n{draft}"
            )})
            try:
                turn = session.turn(critic, use_tools=False, phase=f"reflect:{iteration}")
            except StepLimitExceeded:
                return finish("step_limit_exceeded", draft)
            if any(item.get("type") == "function_call" for item in turn.output):
                return finish("unexpected_tool_call", draft)
            if not turn.output_text.strip():
                return finish("empty_response", draft)
            try:
                review = parse_json_response(turn.output_text)
                if (not isinstance(review, dict) or type(review.get("approved")) is not bool
                        or not isinstance(review.get("feedback"), str)
                        or (not review["approved"] and not review["feedback"].strip())):
                    raise ValueError("review requires approved and concrete feedback")
            except (ValueError, TypeError):
                return finish("invalid_reflection", draft)
            reviews.append({"iteration": iteration, "approved": review["approved"],
                            "feedback": review["feedback"]})
            session.event("reflection", **reviews[-1])
            if review["approved"]:
                approved = True
                return finish("completed", draft)
            session.inputs.append({"role": "user", "content": (
                f"Revise the current answer using this review feedback:\n{review['feedback']}\n"
                "Use tools when new evidence is needed. Keep the same task and permissions. "
                "Provide a complete revised answer, retaining supported evidence and uncertainty."
            )})
            status, revised = session.run_tools(phase=f"refine:{iteration}")
            if status != "completed":
                return finish(status, draft)
            draft = revised
        return finish("completed", draft, limit_reached=bool(reviews))
