"""One bounded run: history, permissions, tool dispatch and audit events.

Strategies share a session so planning and criticism cannot reset the budget.
The default strategy preserves existing result and trace formats.
"""
from __future__ import annotations
import copy
import json
import time
from typing import Any
import uuid
from reclab_agent.tools.base import Toolbox
from reclab_agent.core.config import AgentConfig
from reclab_agent.core.llm import ModelBackend, ModelTurn
from reclab_agent.core.trace import TraceWriter, evidence_paths


SYSTEM_INSTRUCTIONS = """You are RecLab Agent, a recommendation experiment analyst.
Use tools for every numeric or artifact claim. Treat tool output as the only source
of experiment facts. Cite evidence paths returned by tools, distinguish observation
from inference, and stop when the question is answered. Never invent a tool, path,
metric, or experiment result."""


class StepLimitExceeded(Exception):
    """The shared run has no model rounds remaining."""
    request_not_sent = True


def function_calls(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in output if item.get("type") == "function_call"]


def function_tools(toolbox: Toolbox,
                   allowed_tools: set[str] | frozenset[str] | None = None) -> list[dict[str, Any]]:
    """Preserve the existing top-level strict schema conversion."""
    definitions = []
    for spec in toolbox.specs:
        if allowed_tools is not None and spec.name not in allowed_tools:
            continue
        schema = copy.deepcopy(spec.input_schema)
        properties = schema.get("properties", {})
        originally_required = set(schema.get("required", []))
        schema["required"] = list(properties)
        for name, property_schema in properties.items():
            if name not in originally_required and isinstance(property_schema.get("type"), str):
                property_schema["type"] = [property_schema["type"], "null"]
        definitions.append({"type": "function", "name": spec.name,
                            "description": spec.description, "parameters": schema, "strict": True})
    return definitions


class RunSession:
    def __init__(self, toolbox: Toolbox, backend: ModelBackend, config: AgentConfig,
                 trace: TraceWriter, instructions: str, prompt: str, *,
                 task_id: str | None = None,
                 allowed_tools: set[str] | frozenset[str] | None = None,
                 resume_run_id: str | None = None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        known_tools = {spec.name for spec in toolbox.specs}
        if allowed_tools is not None:
            allowed_tools = frozenset(allowed_tools)
            unknown = sorted(allowed_tools - known_tools)
            if unknown:
                raise ValueError(f"allowed_tools contains unknown tools: {unknown}")
            if not allowed_tools:
                raise ValueError("allowed_tools cannot be empty")
        self.toolbox, self.backend, self.config = toolbox, backend, config
        self.trace, self.instructions = trace, instructions
        self.allowed_tools = allowed_tools
        self.tools = function_tools(toolbox, allowed_tools)
        self.inputs: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        self.run_id = resume_run_id or uuid.uuid4().hex
        self.task_id = task_id or self.run_id
        self.started = time.perf_counter()
        self.steps = 0
        self.tool_calls = 0
        self.evidence: set[str] = set()
        self._finished = False
        self.event("run_resumed" if resume_run_id else "run_started", prompt=prompt, max_steps=config.max_steps,
                   allowed_tools=sorted(allowed_tools) if allowed_tools is not None else None)

    def event(self, event: str, **payload: Any) -> None:
        self.trace.write({**payload, "event": event, "run_id": self.run_id, "task_id": self.task_id})

    def _ensure_active(self) -> None:
        if self._finished:
            raise RuntimeError("session already finished")

    def turn(self, instructions: str | None = None, *, use_tools: bool = True,
             phase: str | None = None) -> ModelTurn:
        self._ensure_active()
        if self.steps >= self.config.max_steps:
            raise StepLimitExceeded
        step = self.steps + 1
        for attempt in range(self.config.model_retries + 1):
            try:
                turn = self.backend.respond(
                    instructions=self.instructions if instructions is None else instructions,
                    input_items=self.inputs, tools=self.tools if use_tools else [],
                )
                break
            except Exception as error:
                not_sent = getattr(error, "request_not_sent", False) is True
                self.event("model_error", step=step, attempt=attempt + 1,
                           error_type=type(error).__name__, message=str(error), request_not_sent=not_sent)
                if not_sent or attempt >= self.config.model_retries:
                    raise
        self.steps = step
        self.inputs.extend(turn.output)
        payload: dict[str, Any] = {"step": step, "response_id": turn.response_id,
                                   "function_calls": len(function_calls(turn.output)), "usage": turn.usage,
                                   "output_text": turn.output_text}
        if phase is not None:
            payload["phase"] = phase
        self.event("model_turn", **payload)
        return turn

    def execute(self, turn: ModelTurn) -> None:
        self._ensure_active()
        for call in function_calls(turn.output):
            call_id = call.get("call_id") or call.get("id")
            name = call.get("name", "")
            raw_arguments = call.get("arguments", "{}")
            started = time.perf_counter()
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
                if not isinstance(arguments, dict):
                    raise ValueError("function arguments must decode to an object")
                arguments = {key: value for key, value in arguments.items() if value is not None}
                if self.allowed_tools is not None and name not in self.allowed_tools:
                    envelope = {"ok": False, "tool": name, "error": {
                        "code": "TOOL_NOT_ALLOWED", "message": f"Tool is not allowed for this task: {name}"}}
                else:
                    envelope = self.toolbox.invoke(name, arguments)
            except FileNotFoundError:
                envelope = {"ok": False, "tool": name, "error": {
                    "code": "FILE_NOT_FOUND", "message": "Required local artifact is missing; inspect the recorded source or experiment paths."}}
            except (json.JSONDecodeError, ValueError) as error:
                arguments = None
                envelope = {"ok": False, "tool": name,
                            "error": {"code": "INVALID_JSON_ARGUMENTS", "message": str(error)}}
            self.tool_calls += 1
            paths = evidence_paths(envelope)
            self.evidence.update(paths)
            self.event("tool_call", step=self.steps, call_id=call_id, tool=name,
                       arguments=arguments, result=envelope, evidence_paths=paths,
                       elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
            self.inputs.append({"type": "function_call_output", "call_id": call_id,
                                "output": json.dumps(envelope, ensure_ascii=False)})
        self.checkpoint_conversation()

    def checkpoint_conversation(self):
        from reclab_agent.infrastructure.files import atomic_write
        path = getattr(self, "conversation_path", None)
        if path is not None:
            atomic_write(path, json.dumps(self.inputs, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    def run_tools(self, instructions: str | None = None, *,
                  phase: str | None = None) -> tuple[str, str]:
        self._ensure_active()
        while self.steps < self.config.max_steps:
            turn = self.turn(instructions, phase=phase)
            if not function_calls(turn.output):
                return ("completed" if turn.output_text.strip() else "empty_response", turn.output_text)
            self.execute(turn)
        return "step_limit_exceeded", ""

    def finish(self, status: str, answer: str, **extra: Any) -> dict[str, Any]:
        self._ensure_active()
        self._finished = True
        self.checkpoint_conversation()
        result = {**extra, "run_id": self.run_id, "task_id": self.task_id, "status": status,
                  "answer": answer, "steps": self.steps, "tool_calls": self.tool_calls,
                  "evidence_paths": sorted(self.evidence),
                  "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 3)}
        self.event("run_finished", **{key: value for key, value in result.items()
                                     if key not in {"run_id", "task_id"}})
        return result
