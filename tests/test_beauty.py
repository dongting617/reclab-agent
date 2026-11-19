"""Protocol-preservation and leakage tests for the trusted Beauty importer."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reclab_agent.beauty import import_beauty_bundle
from reclab_agent.datasets import popularity_predictions


def prepared_source():
    # Four training-only users keep all five catalog items in training. Held-out
    # targets deliberately have different warm/cold membership in valid/test.
    return {
        "schema_version": 2, "protocol": "sasrec_source_order_train_catalog",
        "dataset": "Amazon Reviews sequential recommendation",
        "source_files": {"reviews_sha256": "1" * 64, "metadata_sha256": "2" * 64},
        "catalog": [{"item_id": item, "raw_item_id": f"i{item}", "title": "item"}
                    for item in range(1, 6)],
        "stats": {"train_catalog_items": 5, "users_in_sample": 4,
                  "warm_validation_users": 2, "warm_test_users": 2,
                  "train_interactions": 8},
        "users": {
            "warm-both": {"train": [1, 1, 2], "valid": 3, "test": 4,
                          "valid_raw_item_id": "i3", "test_raw_item_id": "i4"},
            "warm-valid": {"train": [3, 4], "valid": 5, "test": 0,
                           "valid_raw_item_id": "i5", "test_raw_item_id": "cold-test"},
            "warm-test": {"train": [5, 2], "valid": 0, "test": 1,
                          "valid_raw_item_id": "cold-valid", "test_raw_item_id": "i1"},
            "no-held-out": {"train": [1], "valid": 0, "test": 0,
                            "valid_raw_item_id": "", "test_raw_item_id": ""},
        },
    }


class BeautyImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, source, name="source.json"):
        path = self.root / name
        path.write_text(json.dumps(source), encoding="utf-8")
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def load(self, source=None, name="bundle", **kwargs):
        path, digest = self.write(source or prepared_source(), f"{name}.json")
        return import_beauty_bundle(path, self.root / name, expected_sha256=digest, **kwargs)

    def test_exact_warm_cohorts_complete_histories_and_training_population(self):
        bundle = self.load(expected_valid_users=2, expected_catalog_items=5)
        self.assertEqual(set(bundle.train), set(prepared_source()["users"]))
        self.assertEqual(set(bundle.valid), {"warm-both", "warm-valid"})
        self.assertEqual(bundle.train["warm-both"], [1, 1, 2])
        self.assertEqual(bundle.valid["warm-both"], {"history": [1, 1, 2], "target": 3})
        test_rows = bundle._rows("test")
        self.assertEqual(set(test_rows), {"warm-both", "warm-test"})
        self.assertEqual(test_rows["warm-both"], {"history": [1, 1, 2, 3], "target": 4})
        self.assertEqual(test_rows["warm-test"], {"history": [5, 2], "target": 1})

    def test_exclusion_reasons_and_exact_source_provenance(self):
        bundle = self.load()
        counts = bundle.manifest["counts"]
        self.assertEqual(counts["valid_excluded"],
                         {"cold_target_outside_train_catalog": 1, "no_held_out_target": 1})
        private = json.loads((bundle.root / "private/provenance.json").read_text())
        self.assertEqual(private["public_data_hash"], bundle.data_hash)
        self.assertEqual(private["prepared_source_sha256"],
                         hashlib.sha256((self.root / "bundle.json").read_bytes()).hexdigest())
        self.assertEqual(private["test_excluded"], counts["valid_excluded"])
        self.assertEqual(private["raw_source_sha256"], prepared_source()["source_files"])
        self.assertTrue(private["validation_did_not_consult_test"])

    def test_test_mutation_cannot_change_public_identity_membership_or_predictions(self):
        first = self.load(name="first")
        changed = prepared_source()
        changed["users"]["warm-both"]["test"] = 5
        changed["users"]["warm-valid"]["test"] = 2
        changed["users"]["warm-test"]["test"] = 0
        changed["users"]["warm-test"]["test_raw_item_id"] = "new-cold-test"
        second = self.load(changed, name="second")
        self.assertEqual(first.data_hash, second.data_hash)
        self.assertEqual(first.dataset_id, second.dataset_id)
        self.assertEqual(first.valid, second.valid)
        self.assertEqual(popularity_predictions(first, first.evaluation_inputs("valid")),
                         popularity_predictions(second, second.evaluation_inputs("valid")))
        for name in ("manifest.json", "train.json", "valid.json"):
            self.assertEqual((first.public / name).read_bytes(), (second.public / name).read_bytes())
        first_origin = json.loads((first.root / "private/provenance.json").read_text())
        second_origin = json.loads((second.root / "private/provenance.json").read_text())
        self.assertNotEqual(first_origin["prepared_source_sha256"], second_origin["prepared_source_sha256"])
        self.assertNotEqual(first_origin["test_rows_sha256"], second_origin["test_rows_sha256"])

    def test_public_files_do_not_copy_test_labels_source_stats_or_original_source(self):
        source = prepared_source()
        source["stats"]["sentinel-test-private"] = "do-not-expose"
        source["users"]["warm-valid"]["test_raw_item_id"] = "private-test-item"
        bundle = self.load(source)
        files = sorted(path.name for path in bundle.public.iterdir())
        self.assertEqual(files, ["manifest.json", "train.json", "valid.json"])
        for path in bundle.public.iterdir():
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("private-test-item", text)
            self.assertNotIn("do-not-expose", text)
            self.assertNotIn("test_raw_item_id", text)
            self.assertNotIn("warm_test_users", text)

    def test_pinned_file_hash_is_mandatory_and_checked_before_writing(self):
        path, digest = self.write(prepared_source())
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            import_beauty_bundle(path, self.root / "new", expected_sha256="a" * 64)
        self.assertFalse((self.root / "new").exists())
        for value in ("", "x" * 64, "A" * 64, True):
            with self.subTest(digest=value), self.assertRaises(ValueError):
                import_beauty_bundle(path, self.root / "new", expected_sha256=value)

    def test_repeated_evaluation_targets_fail_instead_of_filtering_users(self):
        for split, target in (("valid", 1), ("test", 3)):
            source = prepared_source()
            source["users"]["warm-both"][split] = target
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "repeated"):
                self.load(source, name=split)
            self.assertFalse((self.root / split).exists())

    def test_bad_schema_catalog_history_targets_and_counts_are_not_repaired(self):
        changes = [
            ("schema", lambda s: s.update(schema_version=True)),
            ("protocol", lambda s: s.update(protocol="sampled_catalog")),
            ("catalog", lambda s: s["catalog"][0].update(item_id=2)),
            ("history", lambda s: s["users"]["warm-both"].update(train=[True, 2])),
            ("bool-target", lambda s: s["users"]["warm-both"].update(valid=True)),
            ("unknown-target", lambda s: s["users"]["warm-both"].update(valid=999)),
            ("negative-target", lambda s: s["users"]["warm-both"].update(valid=-1)),
            ("counts", lambda s: s["stats"].update(warm_validation_users=1)),
            ("train-counts", lambda s: s["stats"].update(train_interactions=7)),
            ("raw-hash", lambda s: s["source_files"].update(reviews_sha256="bad")),
            ("cold-reason", lambda s: s["users"]["warm-test"].pop("valid_raw_item_id")),
        ]
        for name, mutate in changes:
            source = prepared_source()
            mutate(source)
            with self.subTest(case=name), self.assertRaises(ValueError):
                self.load(source, name=name)
            self.assertFalse((self.root / name).exists())

    def test_catalog_is_strictly_training_derived(self):
        source = prepared_source()
        source["catalog"].append({"item_id": 6})
        source["stats"]["train_catalog_items"] = 6
        with self.assertRaisesRegex(ValueError, "training interactions only"):
            self.load(source)

    def test_frozen_expected_population_is_checked(self):
        for kwargs in ({"expected_valid_users": 3}, {"expected_catalog_items": 6},
                       {"expected_valid_users": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load(**kwargs)

    def test_changed_training_or_validation_content_changes_bundle_identity(self):
        first = self.load(name="first")
        changed = prepared_source()
        changed["users"]["warm-both"]["train"] = [2, 1, 1]
        second = self.load(changed, name="second")
        self.assertNotEqual(first.data_hash, second.data_hash)
        self.assertNotEqual(first.dataset_id, second.dataset_id)

    def test_existing_and_partial_destinations_are_never_overwritten(self):
        bundle = self.load()
        path, digest = self.write(prepared_source())
        original = (bundle.public / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            import_beauty_bundle(path, bundle.root, expected_sha256=digest)
        self.assertEqual((bundle.public / "manifest.json").read_bytes(), original)
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            import_beauty_bundle(path, empty, expected_sha256=digest)
        self.assertEqual(list(empty.iterdir()), [])

    def test_duplicate_json_keys_are_rejected(self):
        path = self.root / "duplicate.json"
        path.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "duplicate keys"):
            import_beauty_bundle(path, self.root / "new", expected_sha256=digest)

    def test_imported_popularity_scores_use_original_warm_validation_users(self):
        bundle = self.load()
        predictions = popularity_predictions(bundle, bundle.evaluation_inputs("valid"))
        result = bundle.grade(predictions)
        self.assertEqual(result["metrics"]["users"], 2)
        self.assertEqual(result["metrics"]["recall@20"], 1.0)
        self.assertEqual(set(result["per_user"]), {"warm-both", "warm-valid"})


if __name__ == "__main__":
    unittest.main()
