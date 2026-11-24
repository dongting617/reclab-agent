"""Common dependency injection and run interface for agent strategies."""
from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from pathlib import Path
from typing import Any
from reclab_agent.tools.base import Toolbox
from reclab_agent.core.config import AgentConfig
from reclab_agent.core.llm import ModelBackend
from reclab_agent.core.session import RunSession, SYSTEM_INSTRUCTIONS
from reclab_agent.core.trace import TraceWriter


class Agent(ABC):
    def __init__(self, toolbox: Toolbox, backend: ModelBackend, *,
                 max_steps: int | None = None, model_retries: int | None = None,
                 trace_path: str | Path | None = None, name: str | None = None,
                 system_prompt: str = SYSTEM_INSTRUCTIONS, config: AgentConfig | None = None):
        config = config or AgentConfig()
        self.config = AgentConfig(config.max_steps if max_steps is None else max_steps,
                                  config.model_retries if model_retries is None else model_retries)
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty text")
        self.name = name or type(self).__name__
        self.toolbox, self.backend = toolbox, backend
        self.system_prompt = system_prompt
        self.max_steps, self.model_retries = self.config.max_steps, self.config.model_retries
        self.trace = TraceWriter(trace_path)
        self._history: list[dict[str, Any]] = []

    def _session(self, prompt: str, *, task_id: str | None = None,
                 allowed_tools: set[str] | frozenset[str] | None = None,
                 resume_run_id: str | None = None) -> RunSession:
        session = RunSession(self.toolbox, self.backend, self.config, self.trace,
                             self.system_prompt, prompt, task_id=task_id, allowed_tools=allowed_tools,
                             resume_run_id=resume_run_id)
        conversation = getattr(self.toolbox, "conversation_path", None)
        if conversation is not None:
            import json
            from reclab_agent.infrastructure.files import read_regular_file
            session.conversation_path = Path(conversation)
            if session.conversation_path.exists():
                history = json.loads(read_regular_file(session.conversation_path, max_bytes=16_000_000))
                if not isinstance(history, list) or any(not isinstance(x, dict) for x in history):
                    raise ValueError("invalid saved conversation")
                session.inputs = history
        # Each run starts fresh. History inspection never grants cross-task memory.
        self._history = session.inputs
        return session

    def get_history(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._history)

    def clear_history(self) -> None:
        self._history = []

    @abstractmethod
    def run(self, prompt: str, *, task_id: str | None = None,
            allowed_tools: set[str] | frozenset[str] | None = None) -> dict[str, Any]:
        """Execute one bounded run and return a structured result."""
