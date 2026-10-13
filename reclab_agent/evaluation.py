"""Deterministic metrics over RecLab Agent run results and traces."""
from __future__ import annotations

import json
from pathlib import Path
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
    answer = result.get("answer", "")
    cited_in_answer = {path for path in observed if path in answer}
    latencies = [event.get("elapsed_ms") for event in calls
                 if isinstance(event.get("elapsed_ms"), (int, float))]
    return {
        "completed": result.get("status") == "completed",
        "tool_selection": evaluate_tool_selection(tools, expected_tools),
        "tool_error_rate": tool_errors / len(calls) if calls else 0.0,
        "evidence_support_rate": len(cited & observed) / len(cited) if cited else 1.0,
        "answer_evidence_citation_rate": (
            len(cited_in_answer) / len(observed) if observed else 1.0
        ),
        "policy_violations": sum(
            event.get("result", {}).get("error", {}).get("code") == "TOOL_NOT_ALLOWED"
            for event in calls
        ),
        "tool_latency_ms": {
            "total": sum(latencies),
            "max": max(latencies) if latencies else 0.0,
        },
        "steps": result.get("steps"),
        "tool_calls": result.get("tool_calls"),
    }


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL runtime trace and reject malformed events."""
    source = Path(path)
    events: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise ValueError(f"line {line_number} is not a trace event")
            events.append(event)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load trace: {source}") from error
    if not events:
        raise ValueError("trace is empty")
    return events


def replay_trace(path: str | Path, expected_tools: list[str]) -> dict[str, Any]:
    """Recompute evaluation metrics from a saved trace without calling a model."""
    events = load_trace(path)
    integrity_errors: list[str] = []
    if events[0].get("event") != "run_started":
        integrity_errors.append("trace must start with run_started")
    if events[-1].get("event") != "run_finished":
        integrity_errors.append("trace must end with run_finished")
    run_ids = {event.get("run_id") for event in events}
    if None in run_ids or len(run_ids) != 1:
        integrity_errors.append("all events must share one non-null run_id")
    finished = [event for event in events if event.get("event") == "run_finished"]
    if len(finished) != 1:
        integrity_errors.append("trace must contain exactly one run_finished event")
        result: dict[str, Any] = {}
    else:
        result = {key: value for key, value in finished[0].items()
                  if key not in {"event"}}
    return {
        "trace_valid": not integrity_errors,
        "integrity_errors": integrity_errors,
        "run_id": result.get("run_id"),
        "metrics": evaluate_run(result, events, expected_tools),
    }
