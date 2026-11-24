"""Shared CLI/Web views over persisted requests and configured study sessions."""
from __future__ import annotations

import json
from pathlib import Path

from reclab_agent.domain.configuration import load_task_configuration, read_json_strict
from reclab_agent.domain.contracts import identifier
from reclab_agent.infrastructure.files import confined_path, read_regular_file
from reclab_agent.memory.storage import RunStore
from reclab_agent.core.trace import event_record


class RunCatalog:
    def __init__(self, root, *, session_roots=None):
        self.root = Path(root).absolute()
        self.store = RunStore(self.root / ".reclab" / "state")
        if session_roots is None:
            session_roots = []
            profile = self.root / "reclab.json"
            if profile.exists():
                config_path = read_json_strict(profile, max_bytes=16_384)["task"]
                config = load_task_configuration(confined_path(self.root, config_path))
                session_roots.append(config.output_directory / "sessions")
        self.session_roots = tuple(Path(p).absolute() for p in session_roots)

    def _sessions(self):
        result = {}
        for root in self.session_roots:
            if not root.is_dir() or root.is_symlink():
                continue
            for path in root.iterdir():
                if path.is_dir() and not path.is_symlink() and (path / "report.json").is_file():
                    key = "study-" + identifier(path.name)
                    if key in result and result[key] != path:
                        raise ValueError("ambiguous study session identity")
                    result[key] = path
        return result

    def get(self, run_id):
        identifier(run_id)
        path = self._sessions().get(run_id)
        if path is None:
            return self.store.get(run_id)
        report = read_json_strict(confined_path(path, "report.json"), max_bytes=32_000_000)
        context = report.get("context", {})
        return {"run_id": run_id, "status": report["status"],
                "task": {"task_id": context.get("task_id", path.name),
                         "question": context.get("task_question", "研究会话"),
                         "mode": context.get("agent_mode")},
                "state": {"stages": report.get("stages", []),
                          "budget_usage": report.get("budget_usage", {}),
                          "requires_reconciliation": report.get("requires_reconciliation", False)},
                "report": report, "origin": "study_session", "updated": (path / "report.json").stat().st_mtime}

    def list(self, limit=50):
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("limit must be 1-200")
        rows = [self.get(key) for key in self._sessions()]
        rows.extend(self.store.list(limit))
        return sorted(rows, key=lambda x: x["updated"], reverse=True)[:limit]

    def events(self, run_id, after=0):
        if type(after) is not int or after < 0:
            raise ValueError("event cursor must be a nonnegative integer")
        self.get(run_id)
        path = self._sessions().get(run_id)
        if path is None:
            return self.store.events(run_id, after)
        trace = confined_path(path, "agent.jsonl")
        if not trace.exists():
            return []
        raw = read_regular_file(trace, max_bytes=64_000_000).decode("utf-8")
        lines = raw.splitlines(keepends=True)
        rows = []
        for seq, line in enumerate(lines, 1):
            if seq <= after:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                if seq == len(lines) and not line.endswith("\n"):
                    break  # A writer may still be appending the final event.
                raise ValueError("invalid saved trace event") from None
            payload = event_record({"study_id": path.name, **payload}, seq, payload.get("timestamp"))
            rows.append({"seq": seq, "run_id": run_id, "event": payload.get("event", "trace"), "payload": payload})
        return rows

    def cancel_waiting(self, run_id):
        current = self.get(run_id)
        if current.get("origin") == "study_session" or current["status"] not in {"queued", "waiting_input"}:
            raise ValueError("only queued or waiting tasks can be cancelled here; running jobs need executor confirmation")
        self.store.cancel_waiting(run_id)
        return self.get(run_id)

    def report_view(self, run_id):
        row = self.get(run_id)
        if row["report"] is None:
            return None
        if row["report"].get("kind") == "evidence_reference":
            return {"summary": row["report"]["quality_claim"], "entries": [],
                    "reference_results": row["report"]["results"]}
        from reclab_agent.reporting.model import build_report_view
        path = self._sessions().get(run_id)
        return build_report_view(row["report"], context={"artifact_directory": path} if path else None)


def run_evidence_demo(root):
    """Exercise local evidence tools and persistence, explicitly without an LLM."""
    from reclab_agent.domain.contracts import TaskSpec
    from reclab_agent.evaluation.benchmark import load_tasks, run_reference_suite
    from reclab_agent.tools import EvidenceToolbox
    root = Path(root).absolute()
    catalog = RunCatalog(root)
    task = TaskSpec("evidence-demo", "本地证据工具演示", "bundled-evidence", "reference")
    run_id = catalog.store.create(task)
    catalog.store.update(run_id, "running", {"kind": "evidence_reference", "llm_connected": False})
    try:
        report = run_reference_suite(EvidenceToolbox(root), load_tasks(root / "tasks/dev.json"))
        report.update(kind="evidence_reference", llm_connected=False, gpu_jobs=0,
                      quality_claim="工具参考调用检查，不代表端到端 Agent 或模型收益。")
        for row in report["results"]:
            catalog.store.event(run_id, "reference_check", row)
        status = "completed" if report["passed"] == report["tasks"] else "partial"
        catalog.store.update(run_id, status, {"kind": "evidence_reference"}, report)
    except Exception:
        catalog.store.update(run_id, "failed", {"error": "local evidence demonstration failed"})
        raise
    return catalog.get(run_id)
