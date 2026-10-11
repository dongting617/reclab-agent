"""Stateful tool-calling runtime with structured, replayable traces."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import time
from typing import Any, Protocol
import uuid

from .tools import RecLabToolbox


SYSTEM_INSTRUCTIONS = """You are RecLab Agent, a recommendation experiment analyst.
Use tools for every numeric or artifact claim. Treat tool output as the only source
of experiment facts. Cite evidence paths returned by tools, distinguish observation
from inference, and stop when the question is answered. Never invent a tool, path,
metric, or experiment result."""


@dataclass(frozen=True)
class ModelTurn:
    """Provider-neutral model response consumed by the runtime."""

    output: list[dict[str, Any]]
    output_text: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class ModelBackend(Protocol):
    def respond(self, *, instructions: str, input_items: list[dict[str, Any]],
                tools: list[dict[str, Any]]) -> ModelTurn:
        """Return either function calls, final text, or both."""


def function_tools(toolbox: RecLabToolbox) -> list[dict[str, Any]]:
    """Translate internal tool specs into Responses API function definitions."""
    definitions = []
    for spec in toolbox.specs:
        schema = copy.deepcopy(spec.input_schema)
        properties = schema.get("properties", {})
        originally_required = set(schema.get("required", []))
        schema["required"] = list(properties)
        for name, property_schema in properties.items():
            if name in originally_required:
                continue
            value_type = property_schema.get("type")
            if isinstance(value_type, str):
                property_schema["type"] = [value_type, "null"]
        definitions.append({
            "type": "function",
            "name": spec.name,
            "description": spec.description,
            "parameters": schema,
            "strict": True,
        })
    return definitions


class OpenAIResponsesBackend:
    """Thin optional adapter around the OpenAI Responses API."""

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
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=False,
            store=False,
        )
        output = [item.model_dump(mode="json") for item in response.output]
        usage = response.usage.model_dump(mode="json") if response.usage else None
        return ModelTurn(output=output, output_text=response.output_text or "",
                         response_id=response.id, usage=usage)


class TraceWriter:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _function_calls(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in output if item.get("type") == "function_call"]


def _evidence_paths(envelope: dict[str, Any]) -> list[str]:
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"path", "evidence_path"} and isinstance(child, str):
                    found.add(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(envelope)
    return sorted(found)


class AgentRuntime:
    """Bounded function-calling loop that records every model and tool step."""

    def __init__(self, toolbox: RecLabToolbox, backend: ModelBackend, *,
                 max_steps: int = 8, model_retries: int = 2,
                 trace_path: str | Path | None = None):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if model_retries < 0:
            raise ValueError("model_retries cannot be negative")
        self.toolbox = toolbox
        self.backend = backend
        self.max_steps = max_steps
        self.model_retries = model_retries
        self.trace = TraceWriter(trace_path)

    def run(self, prompt: str, *, task_id: str | None = None) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        run_id = uuid.uuid4().hex
        task_id = task_id or run_id
        started = time.perf_counter()
        input_items: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = function_tools(self.toolbox)
        tool_calls = 0
        evidence: set[str] = set()
        self.trace.write({"event": "run_started", "run_id": run_id,
                          "task_id": task_id, "prompt": prompt, "max_steps": self.max_steps})

        for step in range(1, self.max_steps + 1):
            turn = self._respond_with_retry(input_items, tools, run_id, task_id, step)
            input_items.extend(turn.output)
            calls = _function_calls(turn.output)
            self.trace.write({"event": "model_turn", "run_id": run_id,
                              "task_id": task_id, "step": step,
                              "response_id": turn.response_id,
                              "function_calls": len(calls), "usage": turn.usage})
            if not calls:
                status = "completed" if turn.output_text.strip() else "empty_response"
                result = {
                    "run_id": run_id, "task_id": task_id, "status": status,
                    "answer": turn.output_text, "steps": step,
                    "tool_calls": tool_calls, "evidence_paths": sorted(evidence),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }
                self.trace.write({"event": "run_finished", **result})
                return result

            for call in calls:
                call_id = call.get("call_id") or call.get("id")
                name = call.get("name", "")
                raw_arguments = call.get("arguments", "{}")
                call_started = time.perf_counter()
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                    if not isinstance(arguments, dict):
                        raise ValueError("function arguments must decode to an object")
                    arguments = {key: value for key, value in arguments.items() if value is not None}
                    envelope = self.toolbox.invoke(name, arguments)
                except (json.JSONDecodeError, ValueError) as error:
                    arguments = None
                    envelope = {"ok": False, "tool": name,
                                "error": {"code": "INVALID_JSON_ARGUMENTS", "message": str(error)}}
                tool_calls += 1
                paths = _evidence_paths(envelope)
                evidence.update(paths)
                elapsed_ms = round((time.perf_counter() - call_started) * 1000, 3)
                self.trace.write({"event": "tool_call", "run_id": run_id,
                                  "task_id": task_id, "step": step, "call_id": call_id,
                                  "tool": name, "arguments": arguments, "result": envelope,
                                  "evidence_paths": paths, "elapsed_ms": elapsed_ms})
                input_items.append({"type": "function_call_output", "call_id": call_id,
                                    "output": json.dumps(envelope, ensure_ascii=False)})

        result = {
            "run_id": run_id, "task_id": task_id, "status": "step_limit_exceeded",
            "answer": "", "steps": self.max_steps, "tool_calls": tool_calls,
            "evidence_paths": sorted(evidence),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        self.trace.write({"event": "run_finished", **result})
        return result

    def _respond_with_retry(self, input_items: list[dict[str, Any]],
                            tools: list[dict[str, Any]], run_id: str,
                            task_id: str, step: int) -> ModelTurn:
        for attempt in range(self.model_retries + 1):
            try:
                return self.backend.respond(instructions=SYSTEM_INSTRUCTIONS,
                                            input_items=input_items, tools=tools)
            except Exception as error:
                self.trace.write({"event": "model_error", "run_id": run_id,
                                  "task_id": task_id, "step": step, "attempt": attempt + 1,
                                  "error_type": type(error).__name__, "message": str(error)})
                if attempt >= self.model_retries:
                    raise
        raise AssertionError("unreachable")
