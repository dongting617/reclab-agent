"""Observation-driven tool use, retaining the original Runtime behavior."""
from __future__ import annotations

from ..core.agent import Agent


class ReActAgent(Agent):
    """Let the model select its next action after each tool observation."""

    def run(self, prompt: str, *, task_id: str | None = None,
            allowed_tools: set[str] | frozenset[str] | None = None) -> dict:
        session = self._session(prompt, task_id=task_id, allowed_tools=allowed_tools)
        status, answer = session.run_tools()
        return session.finish(status, answer)
