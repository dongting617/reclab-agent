"""Content-addressed artifacts and durable task/event state (no candidate imports)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any
import uuid
from contextlib import contextmanager

from .contracts import ArtifactRef, TaskSpec, canonical_json, identifier
from .files import atomic_write, confined_path


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_json(self, kind: str, payload: Any, **metadata: Any) -> ArtifactRef:
        identifier(kind)
        raw = canonical_json(payload).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        artifact_id = f"{kind}-{digest}"
        ref = ArtifactRef(artifact_id, kind, digest, len(raw), metadata)
        blob = confined_path(self.root, f"{digest}.json")
        if not blob.exists():
            atomic_write(blob, raw)
        elif hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
            raise ValueError("existing artifact was modified")
        # Provenance may differ for identical bytes; keep it in each returned ref.
        return ref

    def read_json(self, ref: ArtifactRef) -> Any:
        if len(ref.sha256) != 64 or any(c not in "0123456789abcdef" for c in ref.sha256):
            raise ValueError("invalid artifact hash")
        raw = confined_path(self.root, f"{ref.sha256}.json").read_bytes()
        if len(raw) != ref.bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
            raise ValueError("artifact integrity check failed")
        return json.loads(raw)


class RunStore:
    TERMINAL = {"completed", "partial", "failed", "cancelled", "budget_exceeded"}
    STATES = TERMINAL | {"queued", "running", "waiting_input"}

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "runs.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, task TEXT NOT NULL, status TEXT NOT NULL,
                    state TEXT NOT NULL, report TEXT, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    event TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            """)

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create(self, task: TaskSpec, run_id: str | None = None) -> str:
        run_id = identifier(run_id or uuid.uuid4().hex)
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?, 'queued', '{}', NULL, ?)",
                       (run_id, canonical_json(task.to_dict()), time.time()))
        self.event(run_id, "run_created", {"task_id": task.task_id})
        return run_id

    def get(self, run_id: str) -> dict[str, Any]:
        identifier(run_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return {**dict(row), "task": json.loads(row["task"]), "state": json.loads(row["state"]),
                "report": json.loads(row["report"]) if row["report"] else None}

    def update(self, run_id: str, status: str, state: dict[str, Any], report: dict[str, Any] | None = None) -> None:
        if status not in self.STATES:
            raise ValueError("invalid run state")
        # Serialize before acquiring the write lock; reject NaN/non-JSON state.
        state_json = canonical_json(state)
        report_json = canonical_json(report) if report is not None else None
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if current is None:
                raise KeyError(run_id)
            if current["status"] in self.TERMINAL:
                raise ValueError("terminal runs are immutable; create a new run")
            db.execute("UPDATE runs SET status=?, state=?, report=?, updated=? WHERE run_id=?",
                       (status, state_json, report_json, time.time(), run_id))

    def event(self, run_id: str, event: str, payload: dict[str, Any]) -> None:
        self.get(run_id)
        with self.connect() as db:
            db.execute("INSERT INTO events(run_id,event,payload,created) VALUES(?,?,?,?)",
                       (run_id, event, canonical_json(payload), time.time()))

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        self.get(run_id)
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be 1-200")
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT run_id FROM runs ORDER BY updated DESC LIMIT ?", (limit,))]
        return [self.get(run_id) for run_id in ids]
