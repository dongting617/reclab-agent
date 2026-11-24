"""Stable JSONL events understood by the existing offline evaluator."""
from __future__ import annotations
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any


def trace_hash(value: Any) -> str:
    """Hash request content without duplicating potentially sensitive input in events."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def model_error_category(error: Exception) -> str:
    """Classify provider failures without importing a particular SDK."""
    if getattr(error, "request_not_sent", False) is True:
        return "local_rejection"
    name = type(error).__name__.lower()
    status = getattr(error, "status_code", None)
    if isinstance(error, TimeoutError) or "timeout" in name:
        return "timeout"
    if status == 429 or "ratelimit" in name:
        return "rate_limit"
    if status in {401, 403} or "authentication" in name or "permissiondenied" in name:
        return "authentication"
    if isinstance(error, ConnectionError) or "connection" in name:
        return "connection"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(status, int) and status >= 500:
        return "provider_server"
    if isinstance(status, int) and status >= 400:
        return "provider_request"
    return "unknown"


def event_record(event: dict[str, Any], sequence: int, timestamp: str | None) -> dict[str, Any]:
    """Common event envelope; unknown historical fields stay null."""
    row = {key: None for key in ("study_id", "parent_run_id", "run_id", "task_id",
           "experiment_id", "candidate_id", "stage", "role")}
    row.update(event)
    row.update(schema_version=2, seq=sequence, timestamp=timestamp)
    return row


class TraceWriter:
    """One shared writer per trace; append sequence resumes from persisted lines."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path is not None else None
        self._sequence = 0
        self._lock = RLock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with self.path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.endswith("\n"):
                            raise ValueError("trace has an incomplete final event; reconcile before appending")
                        self._sequence += 1

    def write(self, event: dict[str, Any]) -> None:
        if self.path is not None:
            with self._lock:
                sequence = self._sequence + 1
                row = event_record(event, sequence,
                    datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
                raw = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(raw + "\n")
                self._sequence = sequence


def evidence_paths(envelope: dict[str, Any]) -> list[str]:
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
