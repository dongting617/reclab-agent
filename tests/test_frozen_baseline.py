"""No-PyTorch preflight tests; actual GPU reproduction is a separate evidence run."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reproduce_frozen_b1.py"
spec = importlib.util.spec_from_file_location("frozen_b1_exporter", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def metadata():
    return {"dataset_sha256": exporter.FROZEN_DATA_SHA256,
            "model_config": dict(exporter.FROZEN_MODEL_CONFIG),
            "epoch": 20, "seed": 2026, "checkpoint_format_version": 2,
            "best_valid_ndcg@20": exporter.FROZEN_REFERENCE_NDCG20,
            "model": {"placeholder": "no torch needed for metadata validation"}}


class FrozenBaselineTests(unittest.TestCase):
    def test_exporter_import_does_not_require_torch(self):
        self.assertTrue(callable(exporter.reproduce))
        self.assertEqual(exporter.FROZEN_MODEL_CONFIG["hidden_size"], 64)

    def test_verified_bytes_and_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            path.write_bytes(b"reviewed bytes")
            digest = hashlib.sha256(b"reviewed bytes").hexdigest()
            self.assertEqual(exporter.pinned_bytes(path, digest, "fixture"), b"reviewed bytes")
            path.write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "hardcoded frozen pin"):
                exporter.pinned_bytes(path, digest, "fixture")

    def test_frozen_checkpoint_metadata_and_config_must_match(self):
        exporter.verify_checkpoint_metadata(metadata())
        changes = [
            ("data", lambda row: row.update(dataset_sha256="f" * 64)),
            ("config", lambda row: row["model_config"].update(hidden_size=128)),
            ("config-bool", lambda row: row["model_config"].update(num_layers=True)),
            ("config-extra", lambda row: row["model_config"].update(extra=1)),
            ("epoch", lambda row: row.update(epoch=19)),
            ("seed", lambda row: row.update(seed=1)),
            ("format", lambda row: row.update(checkpoint_format_version=1)),
            ("score", lambda row: row.update(**{"best_valid_ndcg@20": 0.99})),
            ("nan-score", lambda row: row.update(**{"best_valid_ndcg@20": float("nan")})),
            ("missing-state", lambda row: row.update(model={})),
        ]
        for name, change in changes:
            row = metadata()
            change(row)
            with self.subTest(case=name), self.assertRaises(ValueError):
                exporter.verify_checkpoint_metadata(row)

    def test_full_history_mask_is_not_truncated_to_model_input(self):
        history = list(range(1, 81)) + [1]
        bundle = SimpleNamespace(valid={"u": {"history": history, "target": 81}})
        visible, seen = exporter.validation_batch(bundle, ["u"])
        self.assertEqual(visible[0], history[-50:])
        self.assertEqual(len(visible[0]), 50)
        self.assertIn((0, 1), seen)  # Item 2 is older than the visible last 50.
        self.assertEqual(len(seen), 80)  # Repeated item 1 is masked once.

    def test_left_padding_and_mask_columns_preserve_original_item_ids(self):
        bundle = SimpleNamespace(valid={"a": {"history": [7, 2]}, "b": {"history": [3]}})
        visible, seen = exporter.validation_batch(bundle, ["a", "b"])
        self.assertEqual(visible[0], [0] * 48 + [7, 2])
        self.assertEqual(visible[1], [0] * 49 + [3])
        self.assertEqual(seen, [(0, 1), (0, 6), (1, 2)])

    def test_missing_or_changed_public_identity_is_rejected(self):
        bad = SimpleNamespace(data_hash="forged hash")
        with self.assertRaisesRegex(ValueError, "public bundle identity"):
            exporter.verify_public_bundle(bad)
        wrong_population = SimpleNamespace(data_hash=exporter.FROZEN_PUBLIC_DATA_HASH,
                                           valid={}, train={}, manifest={"catalog": [1]})
        with self.assertRaisesRegex(ValueError, "population/catalog"):
            exporter.verify_public_bundle(wrong_population)

    def test_output_never_overwrites_existing_even_if_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "already exists"):
                exporter.require_new_output(directory)
            new_path = Path(directory) / "new"
            self.assertEqual(exporter.require_new_output(new_path), new_path)
            self.assertFalse(new_path.exists())

    def test_bad_pin_fails_before_torch_import_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "not-real-data"
            source.write_bytes(b"arbitrary candidate data")
            with patch.dict("sys.modules", {"torch": None}):
                with self.assertRaisesRegex(ValueError, "prepared data SHA256"):
                    exporter.reproduce(prepared_data=source, bundle_root=root / "bundle",
                                       checkpoint=root / "checkpoint", model_source=root / "model",
                                       output=root / "output")
            self.assertFalse((root / "output").exists())

    def test_invalid_resources_fail_before_file_reads(self):
        for settings in ({"device": "cuda:7"}, {"batch_size": 0}, {"batch_size": True}, {"batch_size": 5000}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                exporter.reproduce(prepared_data="missing", bundle_root="missing", checkpoint="missing",
                                   model_source="missing", output="unused", **settings)


if __name__ == "__main__":
    unittest.main()
