"""Public-data and source contracts; never execute SASRec training or Torch."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from reclab_agent.data.datasets import DatasetBundle, create_synthetic_bundle
from reclab_agent.execution.worker import InputTree
from reclab_agent.experiments.workspace import Workspace


BACKEND = Path(__file__).resolve().parents[1] / "backends" / "sasrec"
MODEL_SHA256 = "37dfa3296117a22e19f6dadd2b2b31be5ee9f30c43febf108cd8911dc15ec1cb"


class SASRecBackendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The reviewed data helper has no Torch dependency. Avoid generating a
        # __pycache__ file inside the source tree registered by InputTree.
        spec = importlib.util.spec_from_file_location("sasrec_data_contract", BACKEND / "src/data.py")
        cls.data = importlib.util.module_from_spec(spec)
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            with patch.dict(sys.modules, {"torch": None}):
                spec.loader.exec_module(cls.data)
        finally:
            sys.dont_write_bytecode = previous

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sasrec-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = create_synthetic_bundle(self.root / "dataset")
        self.config = json.loads((BACKEND / "configs/default.json").read_text(encoding="utf-8"))

    def write_public(self, name, value):
        path = self.bundle.public / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def write_config(self, value):
        path = self.root / "config.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_public_identity_matches_existing_dataset_bundle(self):
        loaded = self.data.load_public_data(self.bundle.public)
        self.assertEqual(loaded["data_hash"], DatasetBundle(self.bundle.root).data_hash)
        self.assertEqual(loaded["manifest"], self.bundle.manifest)
        self.assertEqual(loaded["train"], self.bundle.train)
        self.assertEqual(loaded["valid"], self.bundle.valid)
        self.assertEqual(loaded["num_items"], 40)

    def test_training_repetitions_order_and_validation_cohort_are_preserved(self):
        train, valid = copy.deepcopy(self.bundle.train), copy.deepcopy(self.bundle.valid)
        user = next(iter(valid))
        train[user] = [3, 1, 3, 2, 1]
        valid[user]["history"] = list(train[user])
        self.write_public("train.json", train)
        self.write_public("valid.json", valid)
        loaded = self.data.load_public_data(self.bundle.public)
        self.assertEqual(loaded["train"][user], [3, 1, 3, 2, 1])
        self.assertEqual(loaded["valid"], valid)
        self.assertEqual(set(loaded["train"]), set(train))

    def test_loader_reads_only_the_three_public_files(self):
        private_test = self.bundle.root / "private/test.json"
        private_test.write_text("private labels must never be read", encoding="utf-8")
        (self.bundle.public / "test.json").write_text("also must not be read", encoding="utf-8")
        with patch.object(self.data, "_read_json", wraps=self.data._read_json) as read:
            loaded = self.data.load_public_data(self.bundle.public)
        self.assertEqual({call.args[0] for call in read.call_args_list},
                         {self.bundle.public / name for name in ("manifest.json", "train.json", "valid.json")})
        self.assertEqual(loaded["valid"], self.bundle.valid)

    def test_invalid_training_and_catalog_ids_are_rejected(self):
        user = next(iter(self.bundle.train))
        for history in ([True, 2], [0, 2], [41], [1.0, 2]):
            with self.subTest(history=history):
                train = copy.deepcopy(self.bundle.train)
                train[user] = history
                self.write_public("train.json", train)
                with self.assertRaises(ValueError):
                    self.data.load_public_data(self.bundle.public)
        self.write_public("train.json", self.bundle.train)
        for catalog in ([1, 3], [1, True], [2, 1], []):
            with self.subTest(catalog=catalog):
                manifest = {**self.bundle.manifest, "catalog": catalog}
                self.write_public("manifest.json", manifest)
                with self.assertRaises(ValueError):
                    self.data.load_public_data(self.bundle.public)

    def test_invalid_validation_rows_fail_without_dropping_users(self):
        user = next(iter(self.bundle.valid))
        original = self.bundle.valid[user]
        rows = [
            {**original, "target": True},
            {**original, "target": 41},
            {**original, "target": original["history"][0]},
            {**original, "history": original["history"][:-1]},
            {**original, "history": [True] + original["history"][1:]},
            {**original, "test": 35},
        ]
        for row in rows:
            with self.subTest(row=row):
                valid = copy.deepcopy(self.bundle.valid)
                valid[user] = row
                self.write_public("valid.json", valid)
                with self.assertRaises(ValueError):
                    self.data.load_public_data(self.bundle.public)
        for valid in ({}, {"unknown-user": original}):
            self.write_public("valid.json", valid)
            with self.assertRaises(ValueError):
                self.data.load_public_data(self.bundle.public)

    def test_manifest_schema_and_declared_population_are_checked(self):
        bad_manifests = [
            {**self.bundle.manifest, "schema_version": True},
            {**self.bundle.manifest, "task": "ctr"},
            {**self.bundle.manifest, "dataset_id": "../dataset"},
            {**self.bundle.manifest, "counts": {"train_users": len(self.bundle.train) - 1}},
            {**self.bundle.manifest, "counts": {"valid_users": True}},
        ]
        for manifest in bad_manifests:
            with self.subTest(manifest=manifest):
                self.write_public("manifest.json", manifest)
                with self.assertRaises(ValueError):
                    self.data.load_public_data(self.bundle.public)

    def test_duplicate_json_keys_are_rejected_in_data_and_configuration(self):
        self.write_public("train.json", self.bundle.train)
        valid_path = self.bundle.public / "valid.json"
        valid_path.write_text('{"u":{"history":[1],"target":2,"target":3}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.data.load_public_data(self.bundle.public)
        config_path = self.write_config(self.config)
        raw = config_path.read_text(encoding="utf-8")
        config_path.write_text(raw[:-1] + ',"device":"cpu"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.data.load_config(config_path)

    def test_nonfinite_json_numbers_and_float_overflow_are_rejected(self):
        for number in ("NaN", "Infinity", "-Infinity", "1e309"):
            with self.subTest(number=number):
                raw = json.dumps(self.config).replace('"learning_rate": 0.001',
                                                      '"learning_rate": ' + number)
                config_path = self.root / "config.json"
                config_path.write_text(raw, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    self.data.load_config(config_path)
        manifest = {**self.bundle.manifest, "metadata": float("nan")}
        self.write_public("manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.data.load_public_data(self.bundle.public)

    def test_default_b1_configuration_and_explicit_cpu_selection(self):
        config = self.data.load_config(BACKEND / "configs/default.json")
        expected = {"model": {"file": "src/sasrec.py", "class": "SASRec",
                              "kwargs": {"hidden_size": 64, "num_layers": 1,
                                         "num_heads": 1, "dropout": 0.4}},
                    "max_length": 50, "epochs": 30, "patience": 5,
                    "batch_size": 256, "eval_batch_size": 256, "learning_rate": 0.001,
                    "weight_decay": 0.0001, "grad_clip": 5.0, "device": "cuda"}
        self.assertEqual(config, expected)
        self.assertEqual(self.data.load_config(self.write_config({**config, "device": "cpu"}))["device"], "cpu")

    def test_alternative_model_is_selected_entirely_by_configuration(self):
        model = {"file": "src/alternative.py", "class": "AlternativeNextItemModel",
                 "kwargs": {"width": 16, "normalization": "layer", "options": {"dropout": 0.1}}}
        config = {**self.config, "model": model, "max_length": 1}
        loaded = self.data.load_config(self.write_config(config))
        self.assertEqual(loaded["model"], model)
        self.assertEqual(loaded["max_length"], 1)
        self.assertFalse((BACKEND / model["file"]).exists())

    def test_model_configuration_rejects_traversal_and_reserved_kwargs(self):
        original = self.config["model"]
        models = [
            {**original, "file": "../outside.py"},
            {**original, "file": "/src/model.py"},
            {**original, "file": "src/../model.py"},
            {**original, "file": "src\\model.py"},
            {**original, "file": "src//model.py"},
            {**original, "file": "src/./model.py"},
            {**original, "file": "src/model.json"},
            {**original, "class": "module.Model"},
            {**original, "class": "class"},
            {**original, "kwargs": {"num_items": 40}},
            {**original, "kwargs": {"max_length": 20}},
            {**original, "kwargs": {"invalid-key": 1}},
            {**original, "kwargs": {"class": 1}},
        ]
        for model in models:
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.data.load_config(self.write_config({**self.config, "model": model}))

    def test_config_rejects_invalid_ranges_types_and_unknown_keys(self):
        changes = [
            {"epochs": True}, {"max_length": 0}, {"batch_size": 0},
            {"patience": -1}, {"learning_rate": 0}, {"learning_rate": 10**400},
            {"weight_decay": -1}, {"grad_clip": False},
            {"device": "auto"}, {"learning_reate": 0.01},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.data.load_config(self.write_config({**self.config, **change}))
        incomplete = dict(self.config)
        del incomplete["device"]
        with self.assertRaises(ValueError):
            self.data.load_config(self.write_config(incomplete))

    def test_backend_is_self_contained_and_registers_as_a_frozen_source(self):
        source = InputTree.freeze(BACKEND, "source")
        source.verify()
        paths = sorted(path.relative_to(BACKEND).as_posix()
                       for path in BACKEND.rglob("*") if path.is_file())
        self.assertTrue({"src/train.py", "src/predict.py", "src/sasrec.py", "src/runtime.py",
                         "src/data.py", "configs/default.json"} <= set(paths))
        workspace = Workspace(BACKEND, paths)
        self.assertTrue(workspace.validate()["ok"])
        snapshot = workspace.snapshot(self.root / "snapshots")
        self.assertEqual(snapshot["files"], paths)
        frozen_snapshot = InputTree.freeze(snapshot["directory"], "source")
        frozen_snapshot.verify()
        changed = Path(snapshot["directory"]) / "configs/default.json"
        changed.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed"):
            frozen_snapshot.verify()

    def test_initial_model_source_matches_the_frozen_b1_implementation(self):
        actual = hashlib.sha256((BACKEND / "src/sasrec.py").read_bytes()).hexdigest()
        self.assertEqual(actual, MODEL_SHA256)


if __name__ == "__main__":
    unittest.main()
