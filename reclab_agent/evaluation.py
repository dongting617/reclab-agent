"""Deterministic metrics over RecLab Agent run results and traces."""
from __future__ import annotations

from typing import Any


def evaluate_tool_selection(actual: list[str], expected: list[str]) -> dict[str, Any]:
    actual_set, expected_set = set(actual), set(expected)
    true_positive = len(actual_set & expected_set)
    precision = true_positive / len(actual_set) if actual_set else (1.0 if not expected_set else 0.0)
    recall = true_positive / len(expected_set) if expected_set else 1.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": sorted(expected_set - actual_set),
        "unexpected": sorted(actual_set - expected_set),
    }


def evaluate_run(result: dict[str, Any], trace_events: list[dict[str, Any]],
                 expected_tools: list[str]) -> dict[str, Any]:
    calls = [event for event in trace_events if event.get("event") == "tool_call"]
    tools = [event.get("tool", "") for event in calls]
    tool_errors = sum(not event.get("result", {}).get("ok", False) for event in calls)
    cited = set(result.get("evidence_paths", []))
    observed = {path for event in calls for path in event.get("evidence_paths", [])}
    return {
        "completed": result.get("status") == "completed",
        "tool_selection": evaluate_tool_selection(tools, expected_tools),
        "tool_error_rate": tool_errors / len(calls) if calls else 0.0,
        "evidence_support_rate": len(cited & observed) / len(cited) if cited else 1.0,
        "steps": result.get("steps"),
        "tool_calls": result.get("tool_calls"),
    }
