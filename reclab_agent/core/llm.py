"""Model boundary: strategies consume ModelTurn, not provider SDK objects."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import json
import re


def parse_json_response(text: str):
    """Accept JSON or one complete JSON fence; reject prose and ambiguous blocks."""
    text = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    return json.loads(fence.group(1) if fence else text)


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
        request = {"model": self.model, "instructions": instructions, "input": input_items, "store": False}
        if tools:
            request.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        response = self.client.responses.create(**request)
        output = [item.model_dump(mode="json") for item in response.output]
        usage = response.usage.model_dump(mode="json") if response.usage else None
        return ModelTurn(output, response.output_text or "", response.id, usage)
