"""Durable ledger, local Git, scoped recall and bounded provider contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reclab_agent.integrations.tavily import LiteratureService
from reclab_agent.memory.project import Project
from reclab_agent.workflow.resources import RunResources
from reclab_agent.tools.contracts import ToolError


class ProjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = Project(self.root / "project")

    def test_ledger_is_durable_idempotent_and_rejects_changed_evidence(self):
        entry = {"record": {"experiment_id": "exp-1"}, "metrics": {"ndcg@20": .1}}
        self.project.record(entry, "scope-a", self.root, verified=True)
        self.project.record(entry, "scope-a", self.root, verified=True)
        self.project.accept("scope-a", "exp-1")
        restarted = Project(self.project.root)
        self.assertEqual(len(restarted.experiments("scope-a")), 1)
        self.assertEqual(restarted.current("scope-a")["entry"], entry)
        with self.assertRaises(ValueError):
            restarted.record({**entry, "metrics": {"ndcg@20": .9}}, "scope-a", self.root, verified=True)
        with self.assertRaises(ValueError):
            restarted.accept("scope-b", "exp-1")

    def test_negative_experience_survives_restart_and_reindex_is_scoped(self):
        self.project.note("experience", "scope-a", "failed-gradient", "# Gradient clipping\n\nObserved nonfinite logits. Root cause unknown.")
        self.project.note("experience", "scope-b", "other-dataset", "# Gradient clipping\n\nDifferent dataset.")
        restarted = Project(self.project.root)
        matches = restarted.search("scope-a", "gradient clipping")
        self.assertTrue(matches)
        self.assertTrue(all(x["source_relative"] == "experience/failed-gradient.md" for x in matches))
        restarted.note("experience", "scope-a", "fixed-gradient", "# Gradient clipping\n\nCorrected interpretation.", replaces="failed-gradient")
        self.assertTrue(all(x["source_relative"] == "experience/fixed-gradient.md" for x in restarted.search("scope-a", "gradient")))

    def test_verified_reconciliation_preserves_prior_failure_without_changing_identity(self):
        import json
        record = {"experiment_id": "exp-recovered", "source_hash": "a" * 64,
                  "status": "failed", "training_job_id": "same-train", "score": None}
        self.project.record({"record": record}, "scope-a", self.root, verified=False)
        recovered = {"record": {**record, "status": "completed", "score": {"sha256": "b" * 64}},
                     "metrics": {"ndcg@20": .1}}
        self.project.record(recovered, "scope-a", self.root, verified=True)
        self.assertTrue(self.project.experiments()[0]["verified"])
        with self.project.connect() as db:
            audit = json.loads(db.execute("SELECT payload FROM events WHERE kind='experiment_reconciled'").fetchone()[0])
        self.assertEqual(audit["previous"]["record"]["status"], "failed")
        self.assertEqual(audit["verified"], recovered)
        self.project.accept("scope-a", "exp-recovered")
        with self.assertRaises(ValueError):
            self.project.record({**recovered, "metrics": {"ndcg@20": .9}}, "scope-a", self.root, verified=True)

    def test_reconciliation_rejects_different_training_identity(self):
        record = {"experiment_id": "exp-recovered", "source_hash": "a" * 64,
                  "status": "failed", "training_job_id": "original", "score": None}
        self.project.record({"record": record}, "scope-a", self.root, verified=False)
        for change in ({"source_hash": "c" * 64}, {"training_job_id": "new-training"}):
            with self.assertRaises(ValueError):
                self.project.record({"record": {**record, **change, "status": "completed", "score": {"sha256": "b" * 64}}},
                                    "scope-a", self.root, verified=True)

    def test_git_tracks_source_versions_without_touching_input_repository(self):
        import shutil
        if not shutil.which("git"):
            self.skipTest("Git required for project versioning")
        source = self.root / "source/src"
        source.mkdir(parents=True)
        (source / "model.py").write_text("WIDTH = 1\n", encoding="utf-8")
        snapshot = {"directory": str(source.parent), "files": ["src/model.py"], "source_hash": "a" * 64}
        first = self.project.commit_source(snapshot, "first")
        (source / "model.py").write_text("WIDTH = 2\n", encoding="utf-8")
        second = self.project.commit_source({**snapshot, "source_hash": "b" * 64}, "second")
        self.assertNotEqual(first, second)
        self.assertEqual(self.project.git("show", first + ":src/model.py"), "WIDTH = 1")
        self.assertFalse((source.parent / ".git").exists())

    def test_literature_preserves_snippet_coverage_and_budget(self):
        calls = []
        def reserve():
            if calls:
                raise ToolError("BUDGET_EXCEEDED", "limit reached")
            calls.append(True)
        def transport(endpoint, payload, key):
            self.assertEqual(endpoint, "search")
            return {"results": [{"url": "https://arxiv.org/abs/example", "title": "Paper", "content": "Gradient clipping."}]}
        service = LiteratureService(self.project, "scope-a", reserve, transport=transport)
        with patch.dict("os.environ", {"TAVILY_API_KEY": "fixture-secret"}):
            result = service.search("gradient clipping")
            self.assertEqual(result["references"][0]["coverage"], "snippet_only")
            self.assertNotIn("fixture-secret", str(result))
            self.assertTrue(self.project.search("scope-a", "gradient"))
            with self.assertRaises(ToolError):
                service.search("again")

    def test_network_allowance_and_deadline_survive_restart(self):
        from types import SimpleNamespace
        config = SimpleNamespace(max_wall_seconds=30, max_network_requests=1,
                                 task=SimpleNamespace(budget=SimpleNamespace(max_total_job_seconds=20, max_model_turns=5)))
        limits = RunResources(self.root, config)
        limits.network_request()
        restarted = RunResources(self.root, config, resume=True)
        self.assertEqual(restarted.state["deadline"], limits.state["deadline"])
        with self.assertRaises(ToolError):
            restarted.network_request()
