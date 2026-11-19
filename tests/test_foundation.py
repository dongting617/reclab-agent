import json
from pathlib import Path
import tempfile
import unittest

from reclab_agent.contracts import Budget, TaskSpec, content_hash
from reclab_agent.storage import ArtifactStore, RunStore, confined_path
from reclab_agent.workspace import Workspace, create_workspace


class FoundationTests(unittest.TestCase):
    def test_task_validation_and_roundtrip(self):
        task = TaskSpec("case-1", "improve ranking", "tiny", "popularity")
        self.assertEqual(TaskSpec.from_dict(task.to_dict()), task)
        for budget in ({"max_patches": True}, {"max_cost_cny": float("nan")}, {"max_training_jobs": 0}):
            with self.assertRaises(ValueError):
                Budget(**budget)
        with self.assertRaises(TypeError):
            TaskSpec.from_dict({**task.to_dict(), "unexpected": 1})

    def test_store_rejects_corrupt_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            ref = store.put_json("metrics", {"ndcg": 0.5}, split="valid")
            self.assertEqual(store.read_json(ref), {"ndcg": 0.5})
            (Path(directory) / f"{ref.sha256}.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                store.read_json(ref)

    def test_checkpoint_survives_restart_and_terminal_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RunStore(directory)
            run_id = first.create(TaskSpec("t", "q", "d", "b"))
            first.update(run_id, "running", {"patches": 1, "job_id": "j"})
            second = RunStore(directory)
            self.assertEqual(second.get(run_id)["state"]["job_id"], "j")
            second.update(run_id, "partial", {"patches": 1}, {"reason": "budget"})
            with self.assertRaises(ValueError):
                first.update(run_id, "running", {})
            self.assertEqual(len(second.events(run_id)), 1)

    def test_patch_is_stale_checked_allowlisted_and_snapshotted(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = create_workspace(directory, "candidate", {"src/model.py": "x = 1\n"})
            before = workspace.read("src/model.py")
            edit = workspace.replace("src/model.py", "x = 1", "x = 2", before["sha256"])
            self.assertIn("+x = 2", edit["diff"])
            with self.assertRaises(ValueError):
                workspace.replace("src/model.py", "x = 2", "x = 3", before["sha256"])
            with self.assertRaises(ValueError):
                workspace.read("src/evaluator.py")
            snapshot = workspace.snapshot(Path(directory) / "snapshots")
            self.assertEqual(snapshot, workspace.snapshot(Path(directory) / "snapshots"))
            self.assertTrue(workspace.validate()["ok"])
            with self.assertRaises(ValueError):
                confined_path(directory, "../test.json")
            with self.assertRaises(ValueError):
                confined_path(directory, "C:/secrets")

    def test_syntax_failure_is_not_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = create_workspace(directory, "candidate", {"src/model.py": "raise RuntimeError('must never run on host')\n"})
            self.assertTrue(workspace.validate()["ok"])
            before = workspace.read("src/model.py")
            workspace.replace("src/model.py", before["text"], "def broken(:", before["sha256"])
            self.assertFalse(workspace.validate()["ok"])
