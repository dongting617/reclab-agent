"""Shared CLI/Web persistence, idempotency and real session views."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from reclab_agent.application.runs import RunCatalog, run_evidence_demo
from reclab_agent.domain.contracts import TaskSpec

try:
    from fastapi.testclient import TestClient
    from reclab_agent.web.app import create_app
except ImportError:
    TestClient = None

ROOT = Path(__file__).resolve().parents[1]


class RunCatalogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sessions = self.root / "sessions"
        self.session = self.sessions / "example"
        self.session.mkdir(parents=True)
        self.report = {"status": "completed", "context": {"task_id": "t1", "task_question": "shared"},
                       "stages": [{"stage": "analysis"}], "experiments": []}
        (self.session / "report.json").write_text(json.dumps(self.report))
        self.catalog = RunCatalog(self.root, session_roots=[self.sessions])

    def test_reads_actual_session_and_append_only_trace(self):
        row = self.catalog.get("study-example")
        self.assertEqual(row["report"], self.report)
        trace = self.session / "agent.jsonl"
        trace.write_text('{"event":"first"}\n{"event":"writing"')
        self.assertEqual([x["seq"] for x in self.catalog.events("study-example")], [1])
        with trace.open("a") as stream:
            stream.write('}\n')
        self.assertEqual([x["seq"] for x in self.catalog.events("study-example", 1)], [2])
        self.assertEqual(self.catalog.list()[0]["run_id"], "study-example")

    def test_idempotency_survives_reopening_and_rejects_changed_payload(self):
        task = TaskSpec("t1", "question", "data", "base")
        first = self.catalog.store.create(task, request_key="request-1")
        reopened = RunCatalog(self.root, session_roots=[])
        self.assertEqual(reopened.store.create(task, request_key="request-1"), first)
        self.assertEqual(len(reopened.store.events(first)), 1)
        with self.assertRaises(ValueError):
            reopened.store.create(TaskSpec("t1", "changed", "data", "base"), request_key="request-1")

    def test_cancellation_requires_not_started(self):
        run_id = self.catalog.store.create(TaskSpec("t1", "question", "data", "base"))
        self.assertEqual(self.catalog.cancel_waiting(run_id)["status"], "cancelled")
        with self.assertRaises(ValueError):
            self.catalog.cancel_waiting("study-example")
        with self.assertRaises(ValueError):
            self.catalog.get("../outside")
        running = self.catalog.store.create(TaskSpec("running", "question", "data", "base"))
        self.catalog.store.update(running, "running", {"execution_started": True})
        with self.assertRaises(ValueError):
            self.catalog.cancel_waiting(running)
        self.assertEqual(self.catalog.get(running)["status"], "running")

    def test_reference_demo_is_explicit_and_persists(self):
        for folder in ("evidence", "tasks"):
            shutil.copytree(ROOT / folder, self.root / folder)
        row = run_evidence_demo(self.root)
        self.assertEqual(row["status"], "completed")
        self.assertFalse(row["report"]["llm_connected"])
        self.assertEqual(row["report"]["gpu_jobs"], 0)
        self.assertEqual(row["report"]["passed"], 15)


@unittest.skipIf(TestClient is None, "optional service dependencies not installed")
class WebTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.catalog = RunCatalog(self.root, session_roots=[])
        self.client = TestClient(create_app(self.root, catalog=self.catalog))
        self.task = TaskSpec("web-task", "<script>alert(1)</script>", "data", "base").to_dict()

    def test_page_and_assets_are_available_without_gpu(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertIn("RecLab", self.client.get("/").text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/app.css").status_code, 200)
        self.assertEqual(self.client.get("/static/workbench-charts.js").status_code, 200)
        self.assertEqual(self.client.get("/static/workbench.css").status_code, 200)
        self.assertNotIn("workbench-prototype.js", self.client.get("/").text)
        self.assertEqual(self.client.get("/health").json()["execution_mode"], "offline")

    def test_same_submission_is_stored_once_and_can_be_cancelled(self):
        headers = {"Idempotency-Key": "same-request"}
        first = self.client.post("/api/tasks", json=self.task, headers=headers)
        again = self.client.post("/api/tasks", json=self.task, headers=headers)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(first.json()["run_id"], again.json()["run_id"])
        run_id = first.json()["run_id"]
        self.assertEqual(len(self.client.get("/api/runs").json()["runs"]), 1)
        conflict = self.client.post("/api/tasks", json={**self.task, "question": "different"}, headers=headers)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.client.post(f"/api/runs/{run_id}/cancel").json()["status"], "cancelled")
        after = self.catalog.events(run_id)[-2]["seq"]
        response = self.client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(after)})
        self.assertIn("event: end", response.text)
        self.assertNotIn('"event": "run_created"', response.text)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": "-1"}).status_code, 400)

    def test_external_runner_dispatched_once_and_failure_is_visible(self):
        calls = []
        def runner(task, run_id):
            calls.append(run_id)
            raise RuntimeError("private provider details")
        client = TestClient(create_app(self.root, runner, catalog=self.catalog))
        headers = {"Idempotency-Key": "execute-once"}
        row = client.post("/api/tasks", json=self.task, headers=headers).json()
        client.post("/api/tasks", json=self.task, headers=headers)
        saved = client.get("/api/runs/" + row["run_id"]).json()
        self.assertEqual(calls, [row["run_id"]])
        self.assertEqual(saved["status"], "failed")
        self.assertNotIn("private provider details", json.dumps(saved))

    def test_real_report_endpoint_returns_verified_progress_without_rewriting_report(self):
        from tests import test_user_report as fixtures
        fixture = fixtures.UserReportTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        session = self.root / "sessions" / "fixture"
        session.mkdir(parents=True)
        baseline = fixture.entry("baseline", .08, .15, output=session)
        improved = fixture.entry("improved", .10, .16, output=session)
        report = fixture.report()
        report.update(context=fixture.context, experiments=[baseline, improved])
        path = session / "report.json"
        original = json.dumps(report).encode("utf-8")
        path.write_bytes(original)
        catalog = RunCatalog(self.root, session_roots=[session.parent])
        client = TestClient(create_app(self.root, catalog=catalog))
        payload = client.get("/api/runs/study-fixture/report").json()
        point = payload["view"]["progress"]["groups"][0]["points"][0]
        self.assertTrue(point["milestone"])
        self.assertTrue(payload["view"]["improved"])
        self.assertAlmostEqual(point["relative"], 25)
        self.assertEqual(path.read_bytes(), original)
