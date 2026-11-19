"""A single model turn without tool execution."""
from __future__ import annotations

from ..core.agent import Agent


class SimpleAgent(Agent):
    """Answer directly; a tool request from a backend is never executed."""

    def run(self, prompt: str, *, task_id: str | None = None,
            allowed_tools: set[str] | frozenset[str] | None = None) -> dict:
        session = self._session(prompt, task_id=task_id, allowed_tools=allowed_tools)
        turn = session.turn(use_tools=False)
        if any(item.get("type") == "function_call" for item in turn.output):
            return session.finish("unexpected_tool_call", "")
        status = "completed" if turn.output_text.strip() else "empty_response"
        return session.finish(status, turn.output_text)
