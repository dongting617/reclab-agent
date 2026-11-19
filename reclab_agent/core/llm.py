"""Model boundary: strategies consume ModelTurn, not provider SDK objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelTurn:
    output: list[dict[str, Any]]
    output_text: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class ModelBackend(Protocol):
    def respond(self, *, instructions: str, input_items: list[dict[str, Any]],
                tools: list[dict[str, Any]]) -> ModelTurn:
        """Return function requests, text, or both; do not execute tools here."""


class OpenAIResponsesBackend:
    """Optional Responses API adapter; existing callers keep the same interface."""

    def __init__(self, model: str, client: Any | None = None):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "Install the optional OpenAI dependency with: pip install -e '.[openai]'"
                ) from error
            client = OpenAI()
        self.client = client
        self.model = model

    def respond(self, *, instructions: str, input_items: list[dict[str, Any]],
                tools: list[dict[str, Any]]) -> ModelTurn:
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=input_items,
            tools=tools, tool_choice="auto", parallel_tool_calls=False, store=False,
        )
        output = [item.model_dump(mode="json") for item in response.output]
        usage = response.usage.model_dump(mode="json") if response.usage else None
        return ModelTurn(output, response.output_text or "", response.id, usage)
