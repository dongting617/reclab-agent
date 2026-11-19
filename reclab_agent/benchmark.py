"""Deterministic benchmark manifest and reference execution for RecLab tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tools.contracts import ToolError
from .tools.evidence import RecLabToolbox, json_pointer


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ToolError("INVALID_TASK_FILE", f"Cannot load task manifest: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ToolError("INVALID_TASK_SCHEMA", "Task manifest must use schema_version=1")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ToolError("INVALID_TASK_SCHEMA", "Task manifest must contain a non-empty tasks list")
    ids = []
    for task in tasks:
        if not isinstance(task, dict) or not all(key in task for key in ("id", "prompt", "reference_call", "assertion")):
            raise ToolError("INVALID_TASK_SCHEMA", "Each task requires id, prompt, reference_call and assertion")
        ids.append(task["id"])
    if len(ids) != len(set(ids)):
        raise ToolError("DUPLICATE_TASK_ID", "Task ids must be unique")
    return tasks


def _assert_result(envelope: dict[str, Any], assertion: dict[str, Any]) -> tuple[bool, Any]:
    actual = json_pointer(envelope, assertion["pointer"])
    operation = assertion["op"]
    expected = assertion.get("value")
    if operation == "eq":
        passed = actual == expected
    elif operation == "gt":
        passed = actual > expected
    elif operation == "gte":
        passed = actual >= expected
    elif operation == "lt":
        passed = actual < expected
    elif operation == "contains":
        passed = expected in actual
    else:
        raise ToolError("INVALID_ASSERTION", f"Unsupported assertion op: {operation}")
    return passed, actual


def run_reference_suite(toolbox: RecLabToolbox, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for task in tasks:
        call = task["reference_call"]
        envelope = toolbox.invoke(call["name"], call.get("arguments", {}))
        try:
            passed, actual = _assert_result(envelope, task["assertion"])
            error = None
        except (ToolError, KeyError, TypeError) as exc:
            passed, actual, error = False, None, str(exc)
        results.append({"id": task["id"], "passed": passed, "actual": actual,
                        "expected": task["assertion"].get("value"), "error": error})
    passed_count = sum(row["passed"] for row in results)
    return {"tasks": len(results), "passed": passed_count,
            "pass_rate": passed_count / len(results) if results else 0.0,
            "results": results}

