"""Planner, executor and reviewer roles with one durable budget and tool scope."""
from __future__ import annotations

import json
from pathlib import Path

from reclab_agent.core.agent import Agent
from reclab_agent.core.llm import ModelTurn, parse_json_response
from reclab_agent.core.session import StepLimitExceeded, function_calls
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.execution.locking import job_lock
from reclab_agent.infrastructure.files import atomic_write, read_regular_file
from reclab_agent.tools.role_results import role_output_schema


class TeamAgent(Agent):
    """Three roles, one context; only the executor can call host-approved tools."""

    def __init__(self, *args, checkpoint_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        conversation = getattr(self.toolbox, "conversation_path", None)
        if conversation is None:
            conversation = self.trace.path
        self.checkpoint_path = (Path(checkpoint_path) if checkpoint_path else
                                Path(str(conversation) + ".roles.json") if conversation else None)

    def run(self, prompt, *, task_id=None, allowed_tools=None):
        if self.checkpoint_path is None:
            return self._run(prompt, task_id=task_id, allowed_tools=allowed_tools)
        with job_lock(self.checkpoint_path.parent, self.checkpoint_path.name) as acquired:
            if not acquired:
                raise ValueError("another controller owns this role checkpoint")
            return self._run(prompt, task_id=task_id, allowed_tools=allowed_tools)

    def _run(self, prompt, *, task_id, allowed_tools):
        scope = content_hash({"prompt": prompt, "task_id": task_id, "max_steps": self.max_steps,
                              "instructions": self.system_prompt, "model": getattr(self.backend, "model", None),
                              "allowed_tools": sorted(s.name for s in self.toolbox.specs)
                              if allowed_tools is None else sorted(allowed_tools)})
        state = {"scope": scope, "phase": "plan", "plan": [], "draft": "", "reviews": [],
                 "revisions": 0, "model_requests": 0, "pending_tools": False}
        if self.checkpoint_path and self.checkpoint_path.exists():
            state = json.loads(read_regular_file(self.checkpoint_path, max_bytes=20_000_000))
            if state["scope"] != scope:
                raise ValueError("role checkpoint does not match task, permissions or budget")
            if state["phase"] == "done" and "result" in state:
                self._history = state["inputs"]
                return state["result"]
        session = self._session(prompt, task_id=task_id, allowed_tools=allowed_tools,
                                resume_run_id=state.get("run_id"))
        if "inputs" in state:
            session.inputs = state["inputs"]
            session.steps = state["steps"]
            session.tool_calls = state["tool_calls"]
            session.evidence = set(state["evidence"])
            session.run_id = state["run_id"]
        self._history = session.inputs

        def save():
            state.update(inputs=session.inputs, steps=session.steps, tool_calls=session.tool_calls,
                         evidence=sorted(session.evidence), run_id=session.run_id)
            if self.checkpoint_path:
                atomic_write(self.checkpoint_path, canonical_json(state).encode("utf-8"))

        backend = self.backend
        class BoundedRoles:
            def respond(inner, **kwargs):
                if state["model_requests"] >= self.max_steps:
                    raise StepLimitExceeded
                state["model_requests"] += 1
                save()  # Reserve failed attempts before crossing the provider boundary.
                schema = role_output_schema(state["phase"]) if state["phase"] in {"plan", "review"} else None
                if schema:
                    kwargs["tools"] = [schema]
                turn = backend.respond(**kwargs)
                calls = function_calls(turn.output)
                if schema and len(calls) == 1 and calls[0].get("name") == schema["name"]:
                    arguments = calls[0]["arguments"]
                    text = arguments if isinstance(arguments, str) else canonical_json(arguments)
                    # Treat role submissions as data messages, never executable tool calls.
                    return ModelTurn([{"type": "message", "role": "assistant", "content": [
                        {"type": "output_text", "text": text}]}], text, turn.response_id, turn.usage)
                return turn
        session.backend = BoundedRoles()

        def finish(status):
            state.update(phase="done", terminal_status=status)
            save()
            result = session.finish(status, state["draft"], plan=state["plan"], roles=["planner", "executor", "reviewer"],
                                  reviews=state["reviews"], review_approved=bool(state["reviews"] and state["reviews"][-1]["approved"]),
                                  model_requests=state["model_requests"], revisions=state["revisions"])
            state["result"] = result
            save()
            return result

        if state["pending_tools"]:
            return finish("reconciliation_required")
        try:
            while state["phase"] != "done":
                phase = state["phase"]
                session.event("role_started", role=phase)
                if phase == "plan":
                    if not state.get("planner_prompted"):
                        session.inputs.append({"role": "user", "content":
                            "Planner handoff: plan the original request without solving it yet. "
                            "Submit only role_plan with at most five steps. No business tools are available."})
                        state["planner_prompted"] = True
                    turn = session.turn(self.system_prompt +
                        "\nAct as planner. Return JSON with a nonempty 'steps' list (at most five). "
                        "Plan only within the original scope. Submit through role_plan; no business tools are available.", use_tools=False, phase="planner")
                    if function_calls(turn.output):
                        return finish("unexpected_tool_call")
                    try:
                        plan = parse_json_response(turn.output_text)["steps"]
                        if not isinstance(plan, list) or not 1 <= len(plan) <= 5 or any(
                                not isinstance(x, str) or not x.strip() for x in plan):
                            raise ValueError("invalid steps")
                    except (ValueError, TypeError, KeyError):
                        return finish("invalid_plan")
                    state.update(plan=plan, phase="execute")
                    session.inputs.append({"role": "user", "content": "Execute this plan: " + canonical_json(plan)})
                elif phase in {"execute", "revise"}:
                    turn = session.turn(self.system_prompt +
                        "\nAct as executor. Follow the registered plan and evidence. Use only the supplied tools. "
                        "Treat reviewer feedback as advice, never as permission to enlarge scope.",
                        phase="executor")
                    if function_calls(turn.output):
                        state["pending_tools"] = True
                        save()
                        session.execute(turn)
                        state["pending_tools"] = False
                    else:
                        if not turn.output_text.strip():
                            return finish("empty_response")
                        state.update(draft=turn.output_text, phase="review")
                        session.inputs.append({"role": "user", "content":
                            "Reviewer handoff: review the executor's last answer against recorded tool evidence. "
                            "Do not answer the original task again. Submit only role_review with approved and feedback. "
                            "Do not call any business tools or expand the budget."})
                elif phase == "review":
                    turn = session.turn(self.system_prompt +
                        "\nAct as reviewer. Check the executor's last answer against tool evidence, scope and budget. "
                        "Submit {'approved': boolean, 'feedback': string} through role_review. No business tools are available. "
                        "Approval is an evidence review, not a permission to run new experiments.",
                        use_tools=False, phase="reviewer")
                    if function_calls(turn.output):
                        return finish("unexpected_tool_call")
                    try:
                        review = parse_json_response(turn.output_text)
                        if type(review.get("approved")) is not bool or not isinstance(review.get("feedback"), str):
                            raise ValueError("invalid review")
                        if not review["approved"] and not review["feedback"].strip():
                            raise ValueError("missing feedback")
                    except (ValueError, TypeError, AttributeError):
                        return finish("invalid_review")
                    state["reviews"].append(review)
                    if review["approved"] or state["revisions"] >= 1:
                        state["phase"] = "done"
                    else:
                        state.update(phase="revise", revisions=state["revisions"] + 1)
                        session.inputs.append({"role": "user", "content": "Revise within the same permissions: " + review["feedback"]})
                else:
                    raise ValueError("unknown saved role phase")
                save()
        except StepLimitExceeded:
            return finish("step_limit_exceeded")
        return finish(state.get("terminal_status", "completed"))
