"""Task configuration contracts; no models, credentials or jobs are loaded."""
from __future__ import annotations
import copy
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from reclab_agent.domain.contracts import Budget, TaskSpec
from reclab_agent.domain.configuration import load_task_configuration, read_json_strict


def task_configuration_value():
    return {
        "schema_version": 1,
        "task": TaskSpec("task-1", "Optimize development NDCG", "tiny", "baseline",
                         budget=Budget(max_training_jobs=2)).to_dict(),
        "source": {"directory": "../model", "config": "default",
                   "editable_files": ["src/model.py", "configs/default.json"]},
        "data": {"directory": "../dataset", "adapter": "next_item_v1", "expected_hash": "a" * 64},
        "evaluation": {"evaluator": "ranking_v1", "primary_metric": "ndcg@20",
                       "guardrail_metrics": ["recall@20"]},
        "experiment": {"candidate_ids": ["candidate-1"], "seeds": [2026]},
        "agent": {"model_env": "RECLAB_MODEL", "api_key_env": "OPENAI_API_KEY"},
        "execution": {"backend": "docker", "policy": None},
        "output": {"directory": "../results/run-1"},
    }


class TaskConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="task-cfg-", dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config_dir = self.root / "configuration"
        self.config_dir.mkdir()
        self.path = self.config_dir / "task.json"
        self.value = task_configuration_value()

    def write(self, value=None):
        self.path.write_text(json.dumps(self.value if value is None else value), encoding="utf-8")
        return self.path

    def test_paths_are_based_on_configuration_not_the_callers_working_directory(self):
        self.write()
        with patch("os.getcwd", return_value=str(self.root / "unrelated")):
            loaded = load_task_configuration(self.path)
        self.assertEqual(loaded.source_directory, self.root / "model")
        self.assertEqual(loaded.dataset_directory, self.root / "dataset")
        self.assertEqual(loaded.output_directory, self.root / "results/run-1")
        self.assertEqual(loaded.model_config_path, self.root / "model/configs/default.json")
        self.assertEqual(loaded.task.dataset_id, "tiny")
        self.assertEqual(loaded.seeds, (2026,))

    def test_loading_is_immutable_and_does_not_create_assets_or_read_credentials(self):
        self.write()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "credential-never-returned"}):
            loaded = load_task_configuration(self.path)
        self.assertFalse(loaded.source_directory.exists())
        self.assertFalse(loaded.dataset_directory.exists())
        self.assertFalse(loaded.output_directory.exists())
        self.assertEqual(loaded.api_key_env, "OPENAI_API_KEY")
        self.assertNotIn("credential-never-returned", repr(loaded))
        with self.assertRaises(FrozenInstanceError):
            loaded.config_name = "different"

    def test_native_absolute_paths_and_relative_policy_are_supported(self):
        value = copy.deepcopy(self.value)
        value["source"]["directory"] = str(self.root / "absolute-model")
        value["execution"]["policy"] = "policies/worker.json"
        loaded = load_task_configuration(self.write(value))
        self.assertEqual(loaded.source_directory, self.root / "absolute-model")
        self.assertEqual(loaded.execution_policy_path, self.config_dir / "policies/worker.json")

    def test_onboarding_may_leave_data_hash_and_execution_policy_unbound(self):
        value = copy.deepcopy(self.value)
        value["data"]["expected_hash"] = None
        value["evaluation"]["guardrail_metrics"] = []
        loaded = load_task_configuration(self.write(value))
        self.assertIsNone(loaded.expected_data_hash)
        self.assertIsNone(loaded.execution_policy_path)
        self.assertEqual(loaded.guardrail_metrics, ())

    def test_unknown_fields_and_literal_credentials_are_rejected(self):
        changes = [
            (None, "command", "python train.py"),
            ("source", "command", "python train.py"),
            ("agent", "api_key", "must-not-be-accepted"),
            ("evaluation", "import_path", "external.module:score"),
            ("task", "extra", True),
            ("execution", "image", "python:latest"),
        ]
        for section, key, item in changes:
            with self.subTest(section=section, key=key):
                value = copy.deepcopy(self.value)
                (value if section is None else value[section])[key] = item
                with self.assertRaises(ValueError):
                    load_task_configuration(self.write(value))

    def test_missing_sections_and_wrong_object_shapes_are_rejected(self):
        for section in self.value:
            value = copy.deepcopy(self.value)
            del value[section]
            with self.subTest(section=section), self.assertRaises(ValueError):
                load_task_configuration(self.write(value))
        for section in ("source", "data", "agent", "execution", "output", "task"):
            value = copy.deepcopy(self.value)
            value[section] = []
            with self.subTest(section=section), self.assertRaises(ValueError):
                load_task_configuration(self.write(value))

    def test_source_editable_paths_cannot_escape_the_registered_snapshot(self):
        for path in ("../model.py", "/src/model.py", "src/../model.py", "src//model.py",
                     "src\\model.py", "src/model.txt", "private/test.json", "src/./model.py"):
            value = copy.deepcopy(self.value)
            value["source"]["editable_files"] = [path]
            with self.subTest(path=path), self.assertRaises(ValueError):
                load_task_configuration(self.write(value))

    def test_scope_arrays_reject_empty_duplicate_and_malformed_entries(self):
        changes = [
            ("source", "editable_files", []),
            ("source", "editable_files", ["src/m.py", "src/m.py"]),
            ("experiment", "candidate_ids", []),
            ("experiment", "candidate_ids", ["same", "same"]),
            ("experiment", "candidate_ids", ["../bad"]),
            ("experiment", "seeds", []),
            ("experiment", "seeds", [1, 1]),
            ("experiment", "seeds", [True]),
            ("experiment", "seeds", [2**31]),
            ("experiment", "seeds", [-1]),
            ("evaluation", "guardrail_metrics", ["recall@20", "recall@20"]),
        ]
        for section, key, item in changes:
            value = copy.deepcopy(self.value)
            value[section][key] = item
            with self.subTest(section=section, key=key, item=item), self.assertRaises(ValueError):
                load_task_configuration(self.write(value))

    def test_identifiers_hashes_metric_and_environment_names_are_validated(self):
        changes = [
            ("source", "config", "../default"),
            ("data", "adapter", "some.module:load"),
            ("data", "expected_hash", "A" * 64),
            ("data", "expected_hash", "short"),
            ("evaluation", "evaluator", "module:score"),
            ("evaluation", "primary_metric", " "),
            ("agent", "api_key_env", "a-b"),
            ("agent", "model_env", "1MODEL"),
            ("execution", "backend", "shell"),
        ]
        for section, key, item in changes:
            value = copy.deepcopy(self.value)
            value[section][key] = item
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                load_task_configuration(self.write(value))

    def test_schema_and_budget_keep_existing_type_and_range_contracts(self):
        values = []
        for schema in (True, 2, "1"):
            value = copy.deepcopy(self.value)
            value["schema_version"] = schema
            values.append(value)
        for budget in ({"max_training_jobs": True}, {"max_cost_cny": 10**400},
                       {"max_job_seconds": 0}, {"unknown_budget": 1}):
            value = copy.deepcopy(self.value)
            value["task"]["budget"] = budget
            values.append(value)
        value = copy.deepcopy(self.value)
        value["task"]["schema_version"] = True
        values.append(value)
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_task_configuration(self.write(value))

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for raw in ('{"schema_version":1,"schema_version":1}', '{"bad":NaN}',
                    '{"bad":Infinity}', '{"bad":-Infinity}', '{"bad":1e309}'):
            self.path.write_text(raw, encoding="utf-8")
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                load_task_configuration(self.path)
        self.path.write_text('{"valid":1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_json_strict(self.path, max_bytes=1)

    def test_paths_are_literal_and_foreign_windows_paths_are_not_reinterpreted(self):
        for path in ("~/dataset", "bad\npath", "bad\x00path"):
            value = copy.deepcopy(self.value)
            value["data"]["directory"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                load_task_configuration(self.write(value))
        if os.name != "nt":
            value = copy.deepcopy(self.value)
            value["data"]["directory"] = "C:\\datasets\\beauty"
            with self.assertRaisesRegex(ValueError, "Windows path"):
                load_task_configuration(self.write(value))


if __name__ == "__main__":
    unittest.main()
