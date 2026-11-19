"""Provider-independent tool descriptions and stable failures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ToolError(ValueError):
    """A stable, serializable tool failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
