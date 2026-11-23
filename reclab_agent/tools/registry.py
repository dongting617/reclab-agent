"""Explicit tool registration; registration never grants task permissions."""
from __future__ import annotations
import copy
from typing import Any, Callable
from reclab_agent.tools.base import FunctionTool, Tool, Toolbox
from reclab_agent.tools.contracts import ToolError, ToolSpec


class ToolRegistry:
    """Compose existing toolboxes and new tools behind the same small interface.

    The registry manages available implementations. Agents must still check their
    task allowlist at execution time. Unexpected handler errors propagate instead
    of being mislabeled as recoverable model argument errors.
    """

    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    @property
    def specs(self) -> list[ToolSpec]:
        return copy.deepcopy(list(self._specs.values()))

    def _check_specs(self, specs: list[ToolSpec]) -> None:
        seen = set(self._specs)
        for spec in specs:
            if not isinstance(spec, ToolSpec) or not isinstance(spec.name, str) or not spec.name.strip():
                raise ValueError("tools require a ToolSpec with a nonempty name")
            if spec.name in seen:
                raise ValueError(f"tool is already registered: {spec.name}")
            seen.add(spec.name)

    def register(self, tool: Tool) -> ToolRegistry:
        spec = copy.deepcopy(tool.spec)
        self._check_specs([spec])

        def invoke(arguments: dict[str, Any]) -> dict[str, Any]:
            try:
                return {"ok": True, "tool": spec.name, "result": tool.run(arguments)}
            except ToolError as error:
                return {"ok": False, "tool": spec.name,
                        "error": {"code": error.code, "message": str(error)}}
            except (TypeError, ValueError) as error:
                return {"ok": False, "tool": spec.name,
                        "error": {"code": "INVALID_ARGUMENTS", "message": str(error)}}

        self._specs[spec.name] = spec
        self._handlers[spec.name] = invoke
        return self

    def register_function(self, spec: ToolSpec, handler: Callable[..., dict[str, Any]]) -> ToolRegistry:
        return self.register(FunctionTool(spec, handler))

    def include(self, toolbox: Toolbox) -> ToolRegistry:
        specs = copy.deepcopy(toolbox.specs)
        self._check_specs(specs)
        # Preflight every name before adding any entries, including duplicates
        # inside the supplied toolbox. Preserve its existing envelope exactly.
        for spec in specs:
            self._specs[spec.name] = spec
            self._handlers[spec.name] = lambda arguments, name=spec.name: toolbox.invoke(name, arguments)
        return self

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._handlers:
            return {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"}}
        if arguments is not None and not isinstance(arguments, dict):
            return {"ok": False, "tool": name,
                    "error": {"code": "INVALID_ARGUMENTS", "message": "tool arguments must be an object"}}
        return self._handlers[name]({} if arguments is None else arguments)
