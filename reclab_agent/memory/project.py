"""Project resources and a durable, queryable experiment ledger (stdlib only)."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
from reclab_agent.domain.contracts import canonical_json, content_hash, identifier
from reclab_agent.infrastructure.files import (
    atomic_write,
    confined_path,
    read_regular_file,
    validated_directory,
)
from reclab_agent.memory.retrieval import KnowledgeIndex


class Project:
    def __init__(self, root, *, retrieval=None):
        self.retrieval = retrieval or {}
        self._text_models = None
        self.root = validated_directory(root, create=True)
        for directory in ("code", "wiki/experiments", "wiki/literature", "wiki/experience", "artifacts"):
            confined_path(self.root, directory).mkdir(parents=True, exist_ok=True)
        self.database = self.root / "ledger.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(version INTEGER NOT NULL);
                INSERT INTO metadata SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM metadata);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, scope TEXT NOT NULL,
                    entry TEXT NOT NULL, root TEXT NOT NULL, verified INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS accepted(scope TEXT PRIMARY KEY, experiment_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS notes(id TEXT PRIMARY KEY, scope TEXT NOT NULL,
                    kind TEXT NOT NULL, path TEXT NOT NULL, digest TEXT NOT NULL, replaced_by TEXT);
            """)
            if db.execute("SELECT version FROM metadata").fetchone()[0] != 1:
                raise ValueError("unsupported ledger schema")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            with db:
                yield db
        finally:
            db.close()

    def save_run(self, run_id, payload, kind="checkpoint"):
        raw = canonical_json(payload)
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (run_id, raw))
            db.execute("INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)", (run_id, kind, raw, time.time()))

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return json.loads(row[0])

    def event(self, run_id, kind, payload):
        with self.connect() as db:
            db.execute("INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
                       (run_id, kind, canonical_json(payload), time.time()))

    def record(self, entry, scope, root, *, verified):
        eid = entry["record"]["experiment_id"]
        raw = canonical_json(entry)
        with self.connect() as db:
            prior = db.execute("SELECT entry,verified,scope,root FROM experiments WHERE id=?", (eid,)).fetchone()
            if prior and prior[0] != raw:
                previous = json.loads(prior[0])
                if any(previous.get(key) != entry.get(key) for key in ("record", "metrics", "evaluation_hash")):
                    old, new = previous["record"], entry["record"]
                    identity = ("experiment_id", "plan_hash", "candidate_id", "source_hash", "config_hash",
                                "dataset_id", "protocol_hash", "seed", "image", "runtime", "program_validation")
                    promotable = (not prior["verified"] and verified and prior["scope"] == scope
                        and prior["root"] == str(Path(root).absolute())
                        and old.get("status") in {"failed", "timed_out", "cancelled"}
                        and new.get("status") == "completed" and new.get("score")
                        and all(old.get(key) == new.get(key) for key in identity)
                        and all(not old.get(key) or old[key] == new.get(key)
                                for key in ("training_job_id", "prediction_job_id")))
                    if not promotable:
                        raise ValueError("experiment identity already has different evidence")
                    db.execute("INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
                        (eid, "experiment_reconciled", canonical_json({"previous": previous, "verified": entry}), time.time()))
                    db.execute("UPDATE experiments SET entry=?,verified=1 WHERE id=?", (raw, eid))
                return
            db.execute("INSERT OR IGNORE INTO experiments VALUES(?,?,?,?,?)", (eid, scope, raw, str(Path(root).absolute()), int(verified)))

    def experiments(self, scope=None):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM experiments" + (" WHERE scope=?" if scope else "") + " ORDER BY id",
                              (scope,) if scope else ()).fetchall()
        return [{**dict(x), "entry": json.loads(x["entry"])} for x in rows]

    def accept(self, scope, experiment_id):
        with self.connect() as db:
            row = db.execute("SELECT verified,scope FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if row is None or not row["verified"] or row["scope"] != scope:
                raise ValueError("only verified matching experiments may be accepted")
            db.execute("INSERT INTO accepted VALUES(?,?) ON CONFLICT(scope) DO UPDATE SET experiment_id=excluded.experiment_id",
                       (scope, experiment_id))

    def current(self, scope):
        with self.connect() as db:
            row = db.execute("SELECT experiments.* FROM accepted JOIN experiments ON accepted.experiment_id=experiments.id WHERE accepted.scope=?",
                             (scope,)).fetchone()
        return {**dict(row), "entry": json.loads(row["entry"])} if row else None

    def git(self, *args):
        result = subprocess.run(["git", "-c", "core.hooksPath=.reclab-no-hooks", "-c", "commit.gpgsign=false",
                                 "-c", "user.name=RecLab", "-c", "user.email=reclab@localhost",
                                 "-C", str(self.root / "code"), *args], capture_output=True, text=True, timeout=30,
                                env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
        if result.returncode:
            raise ValueError("project Git operation failed: " + args[0] + ": " + result.stderr[-2000:].strip())
        return result.stdout.strip()

    def configure(self, config, scope, inspection, *, initial_commit=None):
        path = self.root / "project.json"
        if path.exists():
            return
        metadata = {"schema_version": 1, "task_id": config.task.task_id, "objective": config.task.question,
                    "model": inspection["source"]["model"], "source": str(config.source_directory),
                    "data": str(config.dataset_directory), "scope": scope, "seed": config.seeds[0],
                    "evaluation": "valid", "initial_commit": initial_commit}
        atomic_write(path, canonical_json(metadata).encode("utf-8"))

    def commit_source(self, snapshot, ref):
        identifier(ref)
        code = self.root / "code"
        if not (code / ".git").exists():
            self.git("init", "-q")
        existing = [p for p in code.rglob("*") if p.is_file() and ".git" not in p.relative_to(code).parts]
        allowed = set(snapshot["files"])
        for path in existing:
            if path.relative_to(code).as_posix() not in allowed:
                path.unlink()
        for name in snapshot["files"]:
            raw = read_regular_file(confined_path(snapshot["directory"], name), max_bytes=1_000_000)
            atomic_write(confined_path(code, name), raw)
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", "RecLab source " + snapshot["source_hash"])
        commit = self.git("rev-parse", "HEAD")
        self.git("update-ref", "refs/reclab/" + ref, commit)
        return commit

    def note(self, kind, scope, note_id, text, *, replaces=None):
        if kind not in {"experiments", "literature", "experience"}:
            raise ValueError("invalid knowledge kind")
        identifier(note_id)
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 256_000:
            raise ValueError("knowledge text must be nonempty and bounded")
        relative = f"{kind}/{note_id}.md"
        path = confined_path(self.root / "wiki", relative)
        raw = text.encode("utf-8")
        with self.connect() as db:
            prior = db.execute("SELECT scope FROM notes WHERE id=?", (note_id,)).fetchone()
            if prior and prior[0] != scope:
                raise ValueError("note belongs to a different comparison scope")
        atomic_write(path, raw)
        with self.connect() as db:
            db.execute("INSERT INTO notes VALUES(?,?,?,?,?,NULL) ON CONFLICT(id) DO UPDATE SET path=excluded.path,digest=excluded.digest",
                       (note_id, scope, kind, relative, hashlib.sha256(raw).hexdigest()))
            if replaces:
                db.execute("UPDATE notes SET replaced_by=? WHERE id=? AND scope=?", (note_id, replaces, scope))
        return {"note_id": note_id, "source": relative, "sha256": hashlib.sha256(raw).hexdigest()}

    def reindex(self, scope, *, embedding_fn=None, reranker=None):
        with self.connect() as db:
            rows = db.execute("SELECT path FROM notes WHERE scope=? AND replaced_by IS NULL ORDER BY id", (scope,)).fetchall()
        index = KnowledgeIndex(self.root / "wiki", embedding_fn=embedding_fn, reranker=reranker)
        index.build([r[0] for r in rows])
        return index

    def search(self, scope, query, k=5, *, mode=None, embedding_fn=None, reranker=None):
        if type(k) is not int or not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        mode = mode or self.retrieval.get("mode", "bm25")
        if mode != "bm25" and embedding_fn is None:
            from reclab_agent.integrations.text_models import TextModels
            if self._text_models is None:
                options = {k: v for k, v in self.retrieval.items() if k not in {"mode", "allow_download"}}
                self._text_models = TextModels(**options, local_files_only=not self.retrieval.get("allow_download", False))
            embedding_fn = self._text_models.embed
            reranker = self._text_models.rerank if mode == "reranked" else reranker
        return self.reindex(scope, embedding_fn=embedding_fn, reranker=reranker).search(query, k=k, mode=mode)


def knowledge_scope(config, inspection):
    return content_hash({"dataset": inspection["dataset"]["data_hash"],
                         "protocol": inspection["evaluation"]["protocol_hash"],
                         "model_interface": inspection["source"]["model"]["class"], "seed": config.seeds[0]})
