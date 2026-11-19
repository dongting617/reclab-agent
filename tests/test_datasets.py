import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reclab_agent.adapters import SourceBackend
from reclab_agent.datasets import DatasetBundle, create_synthetic_bundle, popularity_predictions
from reclab_agent.grading import RankingProtocol, compare_reports, grade_rankings


class DatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "data"
        self.bundle = create_synthetic_bundle(self.root)

    def write(self, relative, value):
        (self.root / relative).write_text(json.dumps(value), encoding="utf-8")

    def test_public_mount_excludes_private_test_labels(self):
        files = {p.relative_to(self.bundle.public).as_posix()
                 for p in self.bundle.public.rglob("*") if p.is_file()}
        self.assertEqual(files, {"manifest.json", "train.json", "valid.json"})
        self.assertTrue((self.root / "private/test.json").is_file())
        info = self.bundle.info()
        self.assertFalse(info["test_labels_exposed"])
        self.assertEqual(info["quality_evidence"], "fixture_only")
        self.assertNotIn("target", info)
        self.assertTrue(all(isinstance(row, list) for row in self.bundle.train.values()))

    def test_prediction_and_public_inspection_do_not_read_private_labels(self):
        original_read = Path.read_text

        def public_only(path, *args, **kwargs):
            if "private" in path.parts:
                raise AssertionError("candidate-side operation read private labels")
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", new=public_only):
            public_bundle = DatasetBundle(self.root)
            public_bundle.info()
            histories = public_bundle.evaluation_inputs("valid")
            first = popularity_predictions(public_bundle, histories)
            for row in public_bundle.valid.values():
                row["target"] = 40  # Rankings must not consult validation targets either.
            second = popularity_predictions(public_bundle, histories)
        self.assertEqual(first, second)

    def test_changing_test_labels_does_not_change_public_identity_or_predictions(self):
        first = popularity_predictions(self.bundle, self.bundle.evaluation_inputs("valid"))
        self.write("private/test.json", {"secret-user": {"history": [], "target": 40}})
        reloaded = DatasetBundle(self.root)
        second = popularity_predictions(reloaded, reloaded.evaluation_inputs("valid"))
        self.assertEqual(self.bundle.data_hash, reloaded.data_hash)
        self.assertEqual(first, second)

    def test_valid_score_matches_independent_full_catalog_grader(self):
        histories = self.bundle.evaluation_inputs("valid")
        predictions = popularity_predictions(self.bundle, histories)
        protocol = RankingProtocol(f"{self.bundle.dataset_id}-{self.bundle.data_hash}", "valid",
                                   tuple(self.bundle.manifest["catalog"]))
        independent = grade_rankings(predictions,
                                     {u: row["target"] for u, row in self.bundle.valid.items()},
                                     histories, protocol)
        self.assertEqual(self.bundle.grade(predictions), independent)
        self.assertEqual(independent["metrics"]["users"], len(histories))
        for user, ranking in predictions.items():
            self.assertEqual(len(ranking), 20)
            self.assertFalse(set(ranking) & set(histories[user]))

    def test_test_inputs_contain_histories_and_trusted_scorer_loads_targets(self):
        histories = self.bundle.evaluation_inputs("test")
        for user in histories:
            self.assertEqual(histories[user], self.bundle.train[user] + [self.bundle.valid[user]["target"]])
        predictions = popularity_predictions(self.bundle, histories)
        report = self.bundle.grade(predictions, "test")
        self.assertEqual(report["protocol"]["split"], "test")
        self.assertEqual(report["metrics"]["users"], len(histories))
        self.assertTrue(all(isinstance(row, list) for row in histories.values()))

    def test_popularity_is_stable_unique_and_masks_seen(self):
        histories = {"new-user": [1, 2, 3, 4, 5]}
        first = popularity_predictions(self.bundle, histories)
        self.assertEqual(first, popularity_predictions(self.bundle, histories))
        self.assertFalse(set(first["new-user"]) & set(histories["new-user"]))
        self.assertEqual(len(first["new-user"]), len(set(first["new-user"])))

    def test_dataset_split_and_changed_dataset_cannot_be_compared(self):
        valid = self.bundle.grade(popularity_predictions(self.bundle, self.bundle.evaluation_inputs("valid")))
        test = self.bundle.grade(popularity_predictions(self.bundle, self.bundle.evaluation_inputs("test")), "test")
        with self.assertRaises(ValueError):
            compare_reports(valid, test)
        manifest = dict(self.bundle.manifest)
        manifest["dataset_id"] = "different-dataset"
        self.write("public/manifest.json", manifest)
        other = DatasetBundle(self.root)
        other_report = other.grade(popularity_predictions(other, other.evaluation_inputs("valid")))
        with self.assertRaises(ValueError):
            compare_reports(valid, other_report)

    def test_invalid_catalog_is_rejected(self):
        original = dict(self.bundle.manifest)
        for catalog in ([], [1, 1], [1, True], [0, 1], [-1, 1], [1, "2"], [1, float("nan")]):
            with self.subTest(catalog=catalog):
                self.write("public/manifest.json", {**original, "catalog": catalog})
                with self.assertRaises(ValueError):
                    DatasetBundle(self.root)

    def test_invalid_or_seen_validation_target_is_rejected(self):
        original = self.bundle.valid
        first_user = next(iter(original))
        for target in (self.bundle.train[first_user][0], 999, True, float("nan")):
            with self.subTest(target=target):
                valid = {u: dict(row) for u, row in original.items()}
                valid[first_user]["target"] = target
                self.write("public/valid.json", valid)
                with self.assertRaises(ValueError):
                    DatasetBundle(self.root)

    def test_validation_history_must_match_frozen_training_history(self):
        valid = {u: dict(row) for u, row in self.bundle.valid.items()}
        first_user = next(iter(valid))
        valid[first_user]["history"] = []
        self.write("public/valid.json", valid)
        with self.assertRaises(ValueError):
            DatasetBundle(self.root)

    def test_validation_rows_must_be_nonempty_mapping(self):
        for valid in ({}, [], None):
            with self.subTest(valid=valid):
                self.write("public/valid.json", valid)
                with self.assertRaises(ValueError):
                    DatasetBundle(self.root)

    def test_missing_users_and_unsupported_split_are_rejected(self):
        predictions = popularity_predictions(self.bundle, self.bundle.evaluation_inputs("valid"))
        predictions.pop(next(iter(predictions)))
        with self.assertRaises(ValueError):
            self.bundle.grade(predictions)
        with self.assertRaises(ValueError):
            self.bundle.evaluation_inputs("train")
        with self.assertRaises(ValueError):
            self.bundle.grade({}, "arbitrary")

    def test_synthetic_fixture_will_not_overwrite_existing_data(self):
        with self.assertRaises(ValueError):
            create_synthetic_bundle(self.root)

    def test_source_backend_commands_are_container_entrypoints(self):
        backend = SourceBackend("tiny-source", ("src/train.py", "src/predict.py", "configs/default.json"),
                                "python:3.11-slim")
        self.assertEqual(backend.training_command(),
                         ("python", "-B", "/workspace/src/train.py", "--data", "/data", "--output", "/output",
                          "--config", "/workspace/configs/default.json"))
        self.assertEqual(backend.prediction_command(),
                         ("python", "-B", "/workspace/src/predict.py", "--data", "/data", "--output", "/output",
                          "--config", "/workspace/configs/default.json", "--model", "/model", "--split", "valid"))
        self.assertEqual(backend.prediction_command(config_name="small"),
                         ("python", "-B", "/workspace/src/predict.py", "--data", "/data", "--output", "/output",
                          "--config", "/workspace/configs/small.json", "--model", "/model", "--split", "valid"))
        with self.assertRaises(ValueError):
            backend.training_command("../secret")
        with self.assertRaises(ValueError):
            backend.prediction_command("test")
        with self.assertRaises(ValueError):
            backend.prediction_command(config_name="../private")
        with self.assertRaises(ValueError):
            SourceBackend("missing-predict", ("src/train.py",), "python:3.11-slim")


if __name__ == "__main__":
    unittest.main()
