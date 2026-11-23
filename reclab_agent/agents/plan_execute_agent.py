"""Explicit planning followed by bounded tool-using execution."""
from __future__ import annotations
import json
from reclab_agent.core.agent import Agent
from reclab_agent.core.session import StepLimitExceeded


class PlanExecuteAgent(Agent):
    """Execute a JSON plan in order under one shared tool and turn budget.

    This is a task decomposition strategy. Its generated steps do not grant
    approval to a research plan or expand the tools supplied by the caller.
    """

    def __init__(self, *args, max_plan_steps: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        if type(max_plan_steps) is not int or max_plan_steps < 1:
            raise ValueError("max_plan_steps must be a positive integer")
        self.max_plan_steps = max_plan_steps

    def run(self, prompt: str, *, task_id: str | None = None,
            allowed_tools: set[str] | frozenset[str] | None = None) -> dict:
        session = self._session(prompt, task_id=task_id, allowed_tools=allowed_tools)
        plan: list[str] = []
        results: list[dict] = []

        def finish(status: str, answer: str = "") -> dict:
            return session.finish(status, answer, plan=plan, step_results=results)

        planner = (self.system_prompt + "\nPlan the user's task before executing it. "
                   "Return only a JSON object with a 'steps' array of nonempty strings. "
                   f"Use at most {self.max_plan_steps} steps. Do not call tools. "
                   "A plan does not authorize additional tools or experiment scope.")
        turn = session.turn(planner, use_tools=False, phase="plan")
        if any(item.get("type") == "function_call" for item in turn.output):
            return finish("unexpected_tool_call")
        if not turn.output_text.strip():
            return finish("empty_response")
        try:
            payload = json.loads(turn.output_text)
            steps = payload.get("steps") if isinstance(payload, dict) else None
            if (not isinstance(steps, list) or not 1 <= len(steps) <= self.max_plan_steps
                    or any(not isinstance(step, str) or not step.strip() for step in steps)):
                raise ValueError("plan must contain a bounded list of nonempty steps")
            plan = [step.strip() for step in steps]
        except (ValueError, TypeError):
            return finish("invalid_plan")
        session.event("plan_created", plan=plan)

        for index, instruction in enumerate(plan, 1):
            session.inputs.append({"role": "user", "content": (
                f"Execute step {index}/{len(plan)} of the task plan: {instruction}\n"
                + json.dumps({"plan": plan, "completed_steps": results}, ensure_ascii=False) + "\n"
                "Use previous observations and completed steps. Report this step's result. "
                "Stay within the original task and available tool permissions."
            )})
            session.event("plan_step_started", plan_step=index, instruction=instruction)
            status, answer = session.run_tools(phase=f"execute:{index}")
            result = {"step": index, "instruction": instruction, "status": status, "answer": answer}
            results.append(result)
            session.event("plan_step_finished", **result)
            if status != "completed":
                return finish(status)

        session.inputs.append({"role": "user", "content": (
            "All plan steps have finished. Synthesize the final answer to the original task "
            "from their results. Cite observed evidence, retain uncertainties, and do not call tools.\n"
            + json.dumps({"plan": plan, "step_results": results}, ensure_ascii=False)
        )})
        try:
            turn = session.turn(use_tools=False, phase="synthesize")
        except StepLimitExceeded:
            return finish("step_limit_exceeded")
        if any(item.get("type") == "function_call" for item in turn.output):
            return finish("unexpected_tool_call")
        status = "completed" if turn.output_text.strip() else "empty_response"
        return finish(status, turn.output_text)
