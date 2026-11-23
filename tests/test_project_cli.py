"""Project CLI operations share services and never require a model or GPU."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from reclab_agent.cli.projects import project_command
from reclab_agent.domain.research import ExperimentRecord
from tests import test_experiment_runner as fixture


class ProjectCommandTests(unittest.TestCase):
    setUp = fixture.ExperimentRunnerTests.setUp

    def args(self, operation, **kwargs):
        return SimpleNamespace(config=self.config.configuration_path, project_operation=operation, **kwargs)

    def test_initialize_then_write_search_and_reindex_knowledge(self):
        result = project_command(self.args("init"))
        self.assertTrue((Path(result["project"]) / "code/.git").is_dir())
        note = self.root / "experience.md"
        note.write_text("# Numerical stability\n\nNonfinite logits require further diagnosis.", encoding="utf-8")
        project_command(self.args("note", input=note, note_id="numerical", replaces=None))
        self.assertTrue(project_command(self.args("search", query="nonfinite logits"))["matches"])
        self.assertGreater(project_command(self.args("reindex"))["chunks"], 0)
        self.assertEqual(project_command(self.args("experiments", experiment_id=None))["experiments"], [])
        with self.assertRaisesRegex(ValueError, "already initialized"):
            project_command(self.args("init"))

    def test_incomplete_historical_import_is_idempotent_and_not_reusable(self):
        record = ExperimentRecord(experiment_id="old-exp", plan_hash=self.plan.plan_hash, candidate_id="candidate-1",
            source_hash="a" * 64, config_hash="b" * 64, dataset_id=self.plan.dataset_id,
            protocol_hash=self.plan.protocol_hash, image=fixture.FIXTURE_IMAGE, seed=42,
            status="failed", failure="historical failure")
        source = self.root / "historical.json"
        source.write_text(json.dumps({"experiments": [{"record": record.to_dict()}]}), encoding="utf-8")
        for _ in range(2):
            result = project_command(self.args("import", input=source))
            self.assertFalse(result["imported"][0]["verified"])
        rows = project_command(self.args("experiments", experiment_id="old-exp"))["experiments"]
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["verified"])
