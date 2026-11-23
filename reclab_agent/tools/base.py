"""Small tool interfaces shared by evidence and experiment integrations."""
from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Any, Callable, Protocol
from reclab_agent.tools.contracts import ToolSpec


class Toolbox(Protocol):
    """Descriptions shown to the model and the host-side execution boundary."""

    @property
    def specs(self) -> list[ToolSpec]: ...

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]: ...


class Tool(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Describe this tool without exposing its implementation to the model."""

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute trusted host code; the registry adds the result envelope."""


class FunctionTool(Tool):
    """Adapt an existing keyword-argument function without rewriting it."""

    def __init__(self, spec: ToolSpec, handler: Callable[..., dict[str, Any]]):
        if not callable(handler):
            raise TypeError("tool handler must be callable")
        self._spec = copy.deepcopy(spec)
        self._handler = handler

    @property
    def spec(self) -> ToolSpec:
        return copy.deepcopy(self._spec)

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._handler(**arguments)
