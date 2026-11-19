"""Task inspection exercises real assets while never importing model code."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from reclab_agent.contracts import TaskSpec
from reclab_agent.datasets import DatasetBundle, create_synthetic_bundle
from reclab_agent.research.configuration import load_task_configuration
from reclab_agent.research.inspection import EvaluatorDescription, TaskAdapterRegistry, inspect_task
from reclab_agent.research_cli import main


class TaskInspectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="inspect-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        (self.source / "src").mkdir(parents=True)
        (self.source / "configs").mkdir()
        self.marker = self.root / "candidate-was-executed"
        dangerous = f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('must not execute')\n"
        (self.source / "src/model.py").write_text(dangerous + "class Model:\n    pass\n", encoding="utf-8")
        for name in ("train.py", "predict.py"):
            (self.source / "src" / name).write_text(dangerous, encoding="utf-8")
        self.model_config = {"model": {"file": "src/model.py", "class": "Model", "kwargs": {}},
                             "max_length": 3, "epochs": 1, "learning_rate": 0.01}
        self.write_model_config(self.model_config)
        self.bundle = create_synthetic_bundle(self.root / "dataset")
        (self.bundle.root / "private/test.json").write_text("private test sentinel; do not parse", encoding="utf-8")
        self.path = self.root / "task.json"
        self.value = {
            "schema_version": 1,
            "task": TaskSpec("inspect-task", "Improve development ranking", self.bundle.dataset_id,
                             "baseline").to_dict(),
            "source": {"directory": "source", "config": "default",
                       "editable_files": ["src/model.py", "configs/default.json"]},
            "data": {"directory": "dataset", "adapter": "next_item_v1",
                     "expected_hash": self.bundle.data_hash},
            "evaluation": {"evaluator": "next_item_ranking_v1", "primary_metric": "ndcg@20",
                           "guardrail_metrics": ["recall@20"]},
            "experiment": {"candidate_ids": ["candidate-1"], "seeds": [42]},
            "agent": {"model_env": "RECLAB_INSPECT_MODEL", "api_key_env": "RECLAB_INSPECT_KEY"},
            "execution": {"backend": "docker", "policy": None},
            "output": {"directory": "results"},
        }

    def write_model_config(self, value):
        (self.source / "configs/default.json").write_text(json.dumps(value), encoding="utf-8")

    def configuration(self, value=None):
        self.path.write_text(json.dumps(self.value if value is None else value), encoding="utf-8")
        return load_task_configuration(self.path)

    def test_valid_assets_are_inspected_without_models_jobs_or_output_creation(self):
        config = self.configuration()
        with patch.dict(sys.modules, {"torch": None}):
            result = inspect_task(config)
        self.assertTrue(result["inputs_valid"])
        self.assertFalse(result["ready_to_run"])
        self.assertFalse(result["model_imported"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["test_labels_read"])
        self.assertFalse(result["source"]["model_constructor_verified"])
        self.assertEqual(result["dataset"]["data_hash"], self.bundle.data_hash)
        self.assertFalse(self.marker.exists())
        self.assertFalse(config.output_directory.exists())
        self.assertIn("src/train.py", result["source"]["readonly_files"])
        self.assertIn("experiment_runner_not_connected", result["pending"])

    def test_invalid_source_syntax_is_reported_without_running_other_assets(self):
        (self.source / "src/model.py").write_text("class Model(:\n", encoding="utf-8")
        result = inspect_task(self.configuration())
        self.assertFalse(result["inputs_valid"])
        self.assertEqual(result["errors"][0]["check"], "source")
        self.assertEqual(result["errors"][0]["error"], "SyntaxError")
        self.assertFalse(self.marker.exists())

    def test_static_model_selection_and_common_present_settings_are_checked(self):
        models = [
            {**self.model_config, "model": {"file": "src/missing.py", "class": "Model", "kwargs": {}}},
            {**self.model_config, "model": {"file": "src/model.py", "class": "Absent", "kwargs": {}}},
            {**self.model_config, "model": {"file": "src/../model.py", "class": "Model", "kwargs": {}}},
            {**self.model_config, "model": {"file": "src/model.py", "class": "Model", "kwargs": {"num_items": 1}}},
            {**self.model_config, "epochs": True},
            {**self.model_config, "learning_rate": 10**400},
            {**self.model_config, "device": "auto"},
        ]
        for value in models:
            with self.subTest(value=value):
                self.write_model_config(value)
                result = inspect_task(self.configuration())
                self.assertFalse(result["inputs_valid"])
                self.assertEqual(result["errors"][0]["check"], "source")
                self.assertFalse(self.marker.exists())

    def test_custom_trainer_settings_are_not_forced_into_the_sasrec_schema(self):
        self.write_model_config({"model": self.model_config["model"], "custom_training_option": "enabled"})
        result = inspect_task(self.configuration())
        self.assertTrue(result["inputs_valid"])
        self.assertEqual(result["source"]["source_config_validation"],
                         "static_model_spec_and_present_common_fields")
        self.assertFalse(result["source"]["candidate_runtime_verified"])

    def test_unregistered_source_files_and_absent_editable_files_are_rejected(self):
        value = copy.deepcopy(self.value)
        value["source"]["editable_files"].append("src/absent.py")
        self.assertFalse(inspect_task(self.configuration(value))["inputs_valid"])
        (self.source / "scorer.py").write_text("raise RuntimeError('not registered')", encoding="utf-8")
        result = inspect_task(self.configuration())
        self.assertFalse(result["inputs_valid"])
        self.assertIn("source snapshot", result["errors"][0]["detail"])

    def test_dataset_id_and_frozen_public_identity_must_match(self):
        for section, key, value in (("task", "dataset_id", "different"),
                                    ("data", "expected_hash", "f" * 64)):
            configuration = copy.deepcopy(self.value)
            configuration[section][key] = value
            with self.subTest(key=key):
                result = inspect_task(self.configuration(configuration))
                self.assertFalse(result["inputs_valid"])
                self.assertEqual(result["errors"][0]["check"], "dataset")

    def test_builtin_adapter_rejects_catalogs_incompatible_with_common_model_scores(self):
        manifest = copy.deepcopy(self.bundle.manifest)
        manifest["catalog"] = [item * 2 for item in manifest["catalog"]]
        train = {user: [item * 2 for item in history] for user, history in self.bundle.train.items()}
        valid = {user: {"history": [item * 2 for item in row["history"]], "target": row["target"] * 2}
                 for user, row in self.bundle.valid.items()}
        for name, value in (("manifest.json", manifest), ("train.json", train), ("valid.json", valid)):
            (self.bundle.public / name).write_text(json.dumps(value), encoding="utf-8")
        # The general DatasetBundle accepts this catalog. The configured common
        # next-item model interface specifically needs contiguous column IDs.
        DatasetBundle(self.bundle.root)
        result = inspect_task(self.configuration())
        self.assertFalse(result["inputs_valid"])
        self.assertIn("contiguous", result["errors"][0]["detail"])

    def test_ambiguous_public_json_and_extra_private_files_are_rejected(self):
        train = self.bundle.public / "train.json"
        original = train.read_text(encoding="utf-8")
        train.write_text('{"user":[1],"user":[2]}', encoding="utf-8")
        result = inspect_task(self.configuration())
        self.assertFalse(result["inputs_valid"])
        self.assertIn("duplicate", result["errors"][0]["detail"])
        train.write_text(original, encoding="utf-8")
        (self.bundle.public / "test.json").write_text("private labels", encoding="utf-8")
        result = inspect_task(self.configuration())
        self.assertFalse(result["inputs_valid"])
        self.assertIn("private/test", result["errors"][0]["detail"])

    def test_unknown_adapter_or_evaluator_names_never_import_modules(self):
        for section, key in (("data", "adapter"), ("evaluation", "evaluator")):
            value = copy.deepcopy(self.value)
            value[section][key] = "candidate.module"
            with self.subTest(section=section):
                result = inspect_task(self.configuration(value))
                self.assertFalse(result["inputs_valid"])
                self.assertIn("not registered", result["errors"][0]["detail"])
                self.assertNotIn("candidate.module", sys.modules)
                self.assertFalse(self.marker.exists())

    def test_registered_evaluator_must_support_all_requested_metrics(self):
        value = copy.deepcopy(self.value)
        value["evaluation"]["primary_metric"] = "accuracy"
        result = inspect_task(self.configuration(value))
        self.assertFalse(result["inputs_valid"])
        self.assertEqual(result["errors"][0]["check"], "evaluation")

    def test_trusted_host_can_register_new_data_and_evaluator_descriptions(self):
        registry, calls = TaskAdapterRegistry(), []

        def data_loader(path):
            calls.append(("dataset", path))
            return DatasetBundle(path)

        def describe(bundle):
            calls.append(("evaluator", bundle.dataset_id))
            return EvaluatorDescription("a" * 64, ("quality",), {"kind": "trusted_test_fixture"})

        registry.register_dataset("custom_data", data_loader)
        registry.register_evaluator("custom_score", describe)
        value = copy.deepcopy(self.value)
        value["data"]["adapter"] = "custom_data"
        value["evaluation"] = {"evaluator": "custom_score", "primary_metric": "quality", "guardrail_metrics": []}
        result = inspect_task(self.configuration(value), registry=registry)
        self.assertTrue(result["inputs_valid"])
        self.assertEqual([item[0] for item in calls], ["dataset", "evaluator"])
        self.assertEqual(result["evaluation"]["metrics"], ["quality"])
        with self.assertRaises(ValueError):
            registry.register_dataset("custom_data", data_loader)

    def test_unpinned_data_and_absent_execution_credentials_remain_pending(self):
        value = copy.deepcopy(self.value)
        value["data"]["expected_hash"] = None
        with patch.dict(os.environ, {"RECLAB_INSPECT_MODEL": "", "RECLAB_INSPECT_KEY": ""}), \
                patch("reclab_agent.research.inspection.importlib.util.find_spec", return_value=None):
            result = inspect_task(self.configuration(value))
        self.assertTrue(result["inputs_valid"])
        self.assertFalse(result["ready_to_run"])
        for reason in ("public_data_identity_not_pinned", "agent_sdk_not_installed", "agent_model_not_configured",
                       "agent_credentials_not_configured", "execution_policy_not_configured"):
            self.assertIn(reason, result["pending"])

    def test_public_bytes_changed_during_dataset_loading_are_rejected(self):
        registry = TaskAdapterRegistry()

        def changing_loader(path):
            bundle = DatasetBundle(path)
            public_file = bundle.public / "train.json"
            public_file.write_text(public_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return bundle

        registry.register_dataset("changing_data", changing_loader)
        value = copy.deepcopy(self.value)
        value["data"]["adapter"] = "changing_data"
        result = inspect_task(self.configuration(value), registry=registry)
        self.assertFalse(result["inputs_valid"])
        self.assertEqual(result["errors"][0]["check"], "dataset")
        self.assertIn("changed", result["errors"][0]["detail"])

    def test_public_bytes_changed_by_evaluator_description_are_rejected(self):
        registry = TaskAdapterRegistry()

        def changing_evaluator(bundle):
            path = bundle.public / "valid.json"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return EvaluatorDescription("a" * 64, ("ndcg@20", "recall@20"))

        registry.register_evaluator("changing_score", changing_evaluator)
        value = copy.deepcopy(self.value)
        value["evaluation"]["evaluator"] = "changing_score"
        result = inspect_task(self.configuration(value), registry=registry)
        self.assertFalse(result["inputs_valid"])
        self.assertEqual(result["errors"][0]["check"], "evaluation")
        self.assertIn("changed", result["errors"][0]["detail"])

    def test_callback_information_cannot_replace_verified_dataset_identity(self):
        registry = TaskAdapterRegistry()

        def misleading_info(path):
            dataset = DatasetBundle(path)
            dataset.info = lambda: {"data_hash": "unverified", "dataset_id": "unverified",
                                    "adapter": "unverified", "public_tree_sha256": "unverified",
                                    "identity_pinned": False}
            return dataset

        registry.register_dataset("host_data", misleading_info)
        value = copy.deepcopy(self.value)
        value["data"]["adapter"] = "host_data"
        result = inspect_task(self.configuration(value), registry=registry)
        self.assertTrue(result["inputs_valid"])
        self.assertEqual(result["dataset"]["dataset_id"], self.bundle.dataset_id)
        self.assertEqual(result["dataset"]["data_hash"], self.bundle.data_hash)
        self.assertTrue(result["dataset"]["identity_pinned"])
        self.assertEqual(result["dataset"]["adapter"], "host_data")
        self.assertNotEqual(result["dataset"]["public_tree_sha256"], "unverified")

    def test_dataset_loader_cannot_redirect_the_registered_public_tree(self):
        other_bundle = create_synthetic_bundle(self.root / "other-dataset")
        registry = TaskAdapterRegistry()
        registry.register_dataset("redirected_data", lambda path: other_bundle)
        value = copy.deepcopy(self.value)
        value["data"]["adapter"] = "redirected_data"
        result = inspect_task(self.configuration(value), registry=registry)
        self.assertFalse(result["inputs_valid"])
        self.assertIn("configured dataset/public", result["errors"][0]["detail"])

    def test_environment_values_are_never_returned_even_when_names_are_swapped(self):
        value = copy.deepcopy(self.value)
        value["agent"]["model_env"] = "RECLAB_INSPECT_KEY"
        with patch.dict(os.environ, {"RECLAB_INSPECT_KEY": "credential-must-not-be-reported"}), \
                patch("reclab_agent.research.inspection.importlib.util.find_spec", return_value=object()):
            result = inspect_task(self.configuration(value))
        self.assertTrue(result["agent_backend"]["model_configured"])
        self.assertTrue(result["agent_backend"]["credentials_configured"])
        self.assertNotIn("credential-must-not-be-reported", json.dumps(result))

    def test_outputs_cannot_overlap_source_or_data_and_are_never_created(self):
        for path in ("source/results", "dataset/results", "source", "."):
            value = copy.deepcopy(self.value)
            value["output"]["directory"] = path
            with self.subTest(path=path):
                result = inspect_task(self.configuration(value))
                self.assertFalse(result["inputs_valid"])
                self.assertEqual(result["errors"][0]["check"], "paths")
        self.assertFalse((self.source / "results").exists())
        self.assertFalse((self.bundle.root / "results").exists())

    def test_cli_valid_inputs_succeeds_while_execution_is_pending(self):
        self.configuration()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["check", "--config", str(self.path)])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(result["ok"])
        self.assertFalse(result["result"]["ready_to_run"])
        self.assertFalse(self.marker.exists())

    def test_cli_malformed_configuration_and_invalid_inputs_are_structured_errors(self):
        self.path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["check", "--config", str(self.path)])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        self.configuration()
        (self.source / "src/model.py").write_text("class Model(:", encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["check", "--config", str(self.path)])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.getvalue())["result"]["inputs_valid"])
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
