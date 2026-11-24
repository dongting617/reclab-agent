"""One bounded run: history, permissions, tool dispatch and audit events.

Single-agent strategies share a session; independent research/audit sessions
use a host-owned shared request budget. Result and trace formats stay stable.
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
from reclab_agent.core.trace import TraceWriter, evidence_paths, trace_hash, model_error_category


SYSTEM_INSTRUCTIONS = """You are RecLab Agent, a recommendation experiment analyst.
Use tools for every numeric or artifact claim. Treat tool output as the only source
of experiment facts. Cite evidence paths returned by tools, distinguish observation
from inference, and stop when the question is answered. Never invent a tool, path,
metric, or experiment result."""


class StepLimitExceeded(Exception):
    """The shared run has no model rounds remaining."""
    request_not_sent = True


def function_calls(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """从模型输出中筛选工具调用请求。"""
    return [item for item in output if item.get("type") == "function_call"]


def function_tools(toolbox: Toolbox,
                   allowed_tools: set[str] | frozenset[str] | None = None) -> list[dict[str, Any]]:
    """按允许列表生成模型可见的工具定义，并转换为严格参数格式。"""
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
    """管理一次运行的上下文、轮次、工具执行与记录。

    主流程：初始化 → run_loop（turn → execute 循环）→ finish。
    turn 保存模型输出，execute 追加工具结果，供下一轮模型读取。
    """

    def __init__(self, toolbox: Toolbox, backend: ModelBackend, config: AgentConfig,
                 trace: TraceWriter, instructions: str, prompt: str, *,
                 task_id: str | None = None,
                 allowed_tools: set[str] | frozenset[str] | None = None,
                 resume_run_id: str | None = None,
                 trace_context: dict[str, Any] | None = None):
        """校验任务和工具范围，初始化对话、计数器及运行标识。"""
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
        self.trace_context = {"study_id": self.run_id, "parent_run_id": None,
                              "role": "agent", **(trace_context or {})}
        self.experiment_id = self.trace_context.get("experiment_id")
        self.model = getattr(backend, "model", None)
        self.task_prompt_hash = trace_hash(prompt)
        self.started = time.perf_counter()
        self.steps = 0
        self.tool_calls = 0
        self.evidence: set[str] = set()
        self._finished = False
        self.event("run_resumed" if resume_run_id else "run_started", prompt=prompt, max_steps=config.max_steps,
                   allowed_tools=sorted(allowed_tools) if allowed_tools is not None else None)

    def event(self, event: str, **payload: Any) -> None:
        """为事件附加运行标识并写入 Trace。"""
        self.trace.write({**self.trace_context, "experiment_id": self.experiment_id,
                          "model": self.model, "task_prompt_hash": self.task_prompt_hash,
                          **payload, "event": event, "run_id": self.run_id, "task_id": self.task_id})

    def _ensure_active(self) -> None:
        """拒绝继续操作已经结束的会话。"""
        if self._finished:
            raise RuntimeError("session already finished")

    def turn(self, instructions: str | None = None, *, use_tools: bool = True,
             phase: str | None = None) -> ModelTurn:
        """检查轮次上限，调用模型（含重试），保存输出并记录用量。"""
        self._ensure_active()
        if self.steps >= self.config.max_steps:
            raise StepLimitExceeded
        step = self.steps + 1
        request = {"instructions": self.instructions if instructions is None else instructions,
                   "input_items": self.inputs, "tools": self.tools if use_tools else []}
        identity = {"prompt_hash": trace_hash(request), "instructions_hash": trace_hash(request["instructions"]),
                    "input_hash": trace_hash(self.inputs), "tool_schema_hash": trace_hash(request["tools"])}
        if phase is not None:
            identity["phase"] = phase
        for attempt in range(self.config.model_retries + 1):
            started = time.perf_counter()
            request_id = uuid.uuid4().hex
            request_info = {**identity, "request_id": request_id, "step": step, "attempt": attempt + 1}
            self.event("model_request_started", **request_info)
            try:
                turn = self.backend.respond(**request)
                break
            except Exception as error:
                not_sent = getattr(error, "request_not_sent", False) is True
                self.event("model_error", **request_info,
                           elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                           error_type=type(error).__name__, error_category=model_error_category(error),
                           message=str(error), request_not_sent=not_sent)
                if not_sent or attempt >= self.config.model_retries:
                    raise
        self.steps = step
        self.inputs.extend(turn.output)
        payload: dict[str, Any] = {**request_info, "response_id": turn.response_id,
                                   "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                                   "function_calls": len(function_calls(turn.output)),
                                   "usage": turn.usage,
                                   "output_text": turn.output_text}
        self.event("model_turn", **payload)
        return turn

    def execute(self, turn: ModelTurn) -> None:
        """解析参数、检查权限并执行工具，将结果追加到对话中。"""
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
            result = envelope.get("result")
            if isinstance(result, dict) and result.get("experiment_id"):
                self.experiment_id = result["experiment_id"]
            experiment_id = (arguments or {}).get("experiment_id", self.experiment_id)
            paths = evidence_paths(envelope)
            self.evidence.update(paths)
            self.event("tool_call", step=self.steps, call_id=call_id, tool=name,
                       arguments=arguments, result=envelope, evidence_paths=paths,
                       experiment_id=experiment_id,
                       elapsed_ms=round((time.perf_counter() - started) * 1000, 3))
            self.inputs.append({"type": "function_call_output", "call_id": call_id,
                                "output": json.dumps(envelope, ensure_ascii=False)})
        self.checkpoint_conversation()

    def checkpoint_conversation(self):
        """配置保存路径时，原子写入当前对话历史。"""
        from reclab_agent.infrastructure.files import atomic_write
        path = getattr(self, "conversation_path", None)
        if path is not None:
            atomic_write(path, json.dumps(self.inputs, ensure_ascii=False, allow_nan=False).encode("utf-8"))

    def run_loop(self, instructions: str | None = None, *,
                 phase: str | None = None) -> tuple[str, str]:
        """循环调用模型和工具，直到模型不再请求工具或达到轮次上限。"""
        self._ensure_active()
        while self.steps < self.config.max_steps:
            turn = self.turn(instructions, phase=phase)
            calls = function_calls(turn.output)
            if not calls:
                return ("completed" if turn.output_text.strip() else "empty_response",
                        turn.output_text)
            self.execute(turn)
        return "step_limit_exceeded", ""

    def finish(self, status: str, answer: str, **extra: Any) -> dict[str, Any]:
        """结束会话、保存历史，返回答案、状态、证据路径及运行统计。"""
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
