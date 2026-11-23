"""Complete source snapshots retain read-only dependencies without edit access."""
from pathlib import Path
import tempfile
import unittest
from reclab_agent.domain.contracts import content_hash
from reclab_agent.execution.worker import InputTree
from reclab_agent.experiments.workspace import Workspace, create_workspace


FILES = {
    "src/model.py": "width = 16\n",
    "src/train.py": "from model import width\n",
    "src/predict.py": "from model import width\n",
    "src/data.py": "item_count = 40\n",
    "configs/default.json": '{"epochs": 1}\n',
}
EDITABLE = ["src/model.py", "configs/default.json"]


class WorkspaceSourceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ws-src-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def workspace(self):
        return create_workspace(self.root, "candidate", FILES, editable_paths=EDITABLE)

    def test_original_constructor_and_factory_keep_the_original_source_hash(self):
        workspace = create_workspace(self.root, "legacy", FILES)
        self.assertEqual(workspace.source_paths, workspace.editable_paths)
        self.assertEqual(workspace.source_paths, frozenset(FILES))
        snapshot = workspace.snapshot(self.root / "snapshots")
        self.assertEqual(snapshot["source_hash"], content_hash(FILES))
        reopened = Workspace(workspace.root, list(FILES))
        self.assertEqual(reopened.snapshot(self.root / "snapshots"), snapshot)

    def test_readonly_dependencies_are_readable_but_not_editable(self):
        workspace = self.workspace()
        before = workspace.read("src/train.py")
        self.assertEqual(before["text"], FILES["src/train.py"])
        self.assertEqual(before["sha256"], content_hash(FILES["src/train.py"]))
        with self.assertRaisesRegex(ValueError, "editable"):
            workspace.replace("src/train.py", "width", "different", before["sha256"])
        self.assertEqual(workspace.read("src/train.py"), before)
        self.assertEqual(workspace.editable_paths, frozenset(EDITABLE))

    def test_editable_files_still_use_hash_checked_patches(self):
        workspace = self.workspace()
        before = workspace.read("src/model.py")
        workspace.replace("src/model.py", "16", "32", before["sha256"])
        self.assertEqual(workspace.read("src/model.py")["text"], "width = 32\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            workspace.replace("src/model.py", "32", "64", before["sha256"])

    def test_snapshot_contains_all_dependencies_and_registers_with_worker(self):
        workspace = self.workspace()
        snapshot = workspace.snapshot(self.root / "snapshots")
        self.assertEqual(snapshot["files"], sorted(FILES))
        self.assertEqual(snapshot["source_hash"], content_hash(FILES))
        for relative, source in FILES.items():
            self.assertEqual((Path(snapshot["directory"]) / relative).read_text(encoding="utf-8"), source)
        frozen = InputTree.freeze(snapshot["directory"], "source")
        frozen.verify()
        self.assertEqual(workspace.editable_paths, frozenset(EDITABLE))

    def test_readonly_changes_affect_snapshot_identity_without_expanding_edits(self):
        workspace = self.workspace()
        before = workspace.snapshot(self.root / "snapshots")
        # Simulate a trusted source update, not an Agent patch.
        (workspace.root / "src/data.py").write_text("item_count = 41\n", encoding="utf-8")
        after = workspace.snapshot(self.root / "snapshots")
        self.assertNotEqual(before["source_hash"], after["source_hash"])
        self.assertEqual(workspace.editable_paths, frozenset(EDITABLE))
        self.assertEqual((Path(before["directory"]) / "src/data.py").read_text(encoding="utf-8"),
                         FILES["src/data.py"])

    def test_readonly_syntax_is_validated_without_execution(self):
        workspace = self.workspace()
        (workspace.root / "src/train.py").write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
        self.assertTrue(workspace.validate()["ok"])
        (workspace.root / "src/train.py").write_text("def broken(:\n", encoding="utf-8")
        result = workspace.validate()
        self.assertFalse(result["ok"])
        self.assertEqual([row["path"] for row in result["errors"]], ["src/train.py"])

    def test_missing_registered_readonly_files_are_errors_and_not_silently_omitted(self):
        workspace = Workspace(self.root / "incomplete", ["src/model.py"],
                              source_paths=["src/model.py", "src/train.py"])
        (workspace.root / "src").mkdir()
        (workspace.root / "src/model.py").write_text("width = 16\n", encoding="utf-8")
        result = workspace.validate()
        self.assertFalse(result["ok"])
        self.assertEqual([row["path"] for row in result["errors"]], ["src/train.py"])
        with self.assertRaises(FileNotFoundError):
            workspace.snapshot(self.root / "snapshots")

    def test_editable_subset_must_be_registered(self):
        with self.assertRaisesRegex(ValueError, "subset"):
            Workspace(self.root / "invalid", ["src/model.py"], source_paths=["src/train.py"])
        with self.assertRaisesRegex(ValueError, "subset"):
            Workspace(self.root / "empty", ["src/model.py"], source_paths=[])
        with self.assertRaisesRegex(ValueError, "editable"):
            Workspace(self.root / "no-edits", [], source_paths=["src/train.py"])

    def test_unregistered_files_remain_inaccessible_and_outside_the_snapshot(self):
        workspace = self.workspace()
        (workspace.root / "src/extra.py").write_text("secret = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            workspace.read("src/extra.py")
        with self.assertRaisesRegex(ValueError, "editable"):
            workspace.replace("src/extra.py", "1", "2", content_hash("secret = 1\n"))
        snapshot = workspace.snapshot(self.root / "snapshots")
        self.assertNotIn("src/extra.py", snapshot["files"])
        self.assertFalse((Path(snapshot["directory"]) / "src/extra.py").exists())

    def test_every_source_path_is_checked_even_when_readonly(self):
        invalid_paths = ["../train.py", "/src/train.py", "C:/src/train.py", "src/../train.py",
                         "src//train.py", "src/./train.py", "src\\train.py",
                         "private/test.json", "src/train.txt"]
        for index, relative in enumerate(invalid_paths):
            with self.subTest(path=relative), self.assertRaises(ValueError):
                Workspace(self.root / f"invalid-{index}", ["src/model.py"],
                          source_paths=["src/model.py", relative])


if __name__ == "__main__":
    unittest.main()
