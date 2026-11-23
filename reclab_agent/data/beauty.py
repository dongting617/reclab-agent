"""Trusted import of the frozen Beauty SASRec train-catalog data protocol.

This adapter does not prepare new splits or repair malformed source records.
Public validation membership is fixed before the separate test-label pass.
Only the returned bundle's ``public`` directory may be mounted for candidates.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.data.datasets import DatasetBundle
from reclab_agent.memory.storage import atomic_write, confined_path


SOURCE_PROTOCOL = "sasrec_source_order_train_catalog"
ADAPTER_VERSION = "beauty-train-catalog-v1"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("source JSON contains duplicate keys")
        result[key] = value
    return result


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_count(actual: int, expected: int | None, name: str) -> None:
    if expected is not None:
        _integer(expected, f"expected_{name}", 1)
        if actual != expected:
            raise ValueError(f"{name} count differs from the frozen expectation: {actual} != {expected}")


def _zero_reason(row: dict[str, Any], split: str) -> str:
    raw = row.get(f"{split}_raw_item_id")
    if not isinstance(raw, str):
        raise ValueError(f"{split}_raw_item_id must be a string for zero targets")
    return "cold_target_outside_train_catalog" if raw else "no_held_out_target"


def import_beauty_bundle(
    source_path: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
    expected_valid_users: int | None = None,
    expected_catalog_items: int | None = None,
) -> DatasetBundle:
    """Import one exact prepared source without changing its evaluation cohort.

    ``expected_sha256`` pins the *source file bytes*, unlike ``data_hash`` which
    binds the canonical public JSON contract. Test values and counts occur only
    in private provenance, so they cannot influence public membership or identity.
    Existing destinations (including empty ones) are deliberately never reused.
    """
    expected_sha256 = _digest(expected_sha256, "expected_sha256")
    source_path = Path(source_path)
    destination = Path(destination)
    # Check the destination through the same reparse/traversal guard as bundles.
    confined_path(destination, "public/manifest.json")
    if destination.exists():
        raise ValueError("dataset destination already exists; use a new frozen destination")
    raw = source_path.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    if source_sha256 != expected_sha256:
        raise ValueError("prepared source SHA256 differs from the frozen expectation")
    source = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(source, dict) or type(source.get("schema_version")) is not int or source["schema_version"] != 2:
        raise ValueError("Beauty adapter requires source schema_version 2")
    if source.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError("Beauty adapter requires sasrec_source_order_train_catalog")
    source_files = source.get("source_files")
    if not isinstance(source_files, dict):
        raise ValueError("source_files must contain raw reviews and metadata SHA256 digests")
    source_files = {key: _digest(source_files.get(key), key)
                    for key in ("reviews_sha256", "metadata_sha256")}
    catalog_rows = source.get("catalog")
    if not isinstance(catalog_rows, list) or not catalog_rows:
        raise ValueError("catalog must be a nonempty ordered item list")
    catalog = []
    for expected_item_id, row in enumerate(catalog_rows, 1):
        if not isinstance(row, dict) or type(row.get("item_id")) is not int or row["item_id"] != expected_item_id:
            raise ValueError("source catalog item IDs must be contiguous and ordered from 1")
        catalog.append(row["item_id"])
    _optional_count(len(catalog), expected_catalog_items, "catalog_items")
    catalog_set = set(catalog)
    users = source.get("users")
    if not isinstance(users, dict) or not users:
        raise ValueError("users must be a nonempty mapping")
    stats = source.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("source stats must be an object")
    if _integer(stats.get("train_catalog_items"), "train_catalog_items", 1) != len(catalog):
        raise ValueError("source stats and actual catalog disagree")
    if _integer(stats.get("users_in_sample"), "users_in_sample", 1) != len(users):
        raise ValueError("source stats and actual user count disagree")

    # Pass one: only source train/valid fields determine public data and cohort.
    train: dict[str, list[int]] = {}
    valid: dict[str, dict[str, Any]] = {}
    valid_excluded = {"cold_target_outside_train_catalog": 0, "no_held_out_target": 0}
    for user in sorted(users):
        row = users[user]
        if not isinstance(user, str) or not user.strip() or not isinstance(row, dict):
            raise ValueError("source users must map nonempty IDs to objects")
        history = row.get("train")
        if not isinstance(history, list) or any(type(item) is not int or item not in catalog_set for item in history):
            raise ValueError("source training history contains invalid items")
        train[user] = list(history)  # Preserve repetitions and complete source order.
        target = _integer(row.get("valid"), "validation target")
        if target == 0:
            valid_excluded[_zero_reason(row, "valid")] += 1
            continue
        if target not in catalog_set:
            raise ValueError("positive validation target is outside the training catalog")
        if target in history:
            raise ValueError("repeated validation target: register a repeat-aware protocol; do not drop this user")
        valid[user] = {"history": list(history), "target": target}
    if not valid:
        raise ValueError("source has no warm validation users")
    _optional_count(len(valid), expected_valid_users, "valid_users")
    if _integer(stats.get("warm_validation_users"), "warm_validation_users") != len(valid):
        raise ValueError("source stats and actual warm validation cohort disagree")
    if _integer(stats.get("train_interactions"), "train_interactions") != sum(map(len, train.values())):
        raise ValueError("source stats and training interactions disagree")
    if {item for history in train.values() for item in history} != catalog_set:
        raise ValueError("catalog must be derived from training interactions only")
    public_protocol = {
        "source_protocol": SOURCE_PROTOCOL,
        "adapter_version": ADAPTER_VERSION,
        "ranking": "full_catalog_single_target",
        "catalog_policy": "train_items_only",
        "history_policy": "complete_train_history_preserve_order_and_repetitions",
        "validation_population": "all_and_only_source_valid_gt_zero",
        "seen_mask": "all_history_items",
        "repeat_target_policy": "raise",
        "ks": [10, 20],
    }
    public_identity = {"protocol": public_protocol, "source_files": source_files,
                       "catalog": catalog, "train": train, "valid": valid}
    public_fingerprint = content_hash(public_identity)
    manifest = {
        "schema_version": 1, "dataset_id": f"beauty-d1-{public_fingerprint[:16]}",
        "profile": "amazon-beauty-full-train-catalog", "task": "next_item",
        "catalog": catalog, "quality_evidence": "real_data_import_only",
        "protocol": public_protocol, "raw_source_sha256": source_files,
        "validation_cohort_sha256": content_hash(sorted(valid)),
        "public_content_sha256": public_fingerprint,
        "counts": {"source_users": len(users), "train_users": len(train),
                   "train_interactions": sum(map(len, train.values())),
                   "valid_users": len(valid), "valid_excluded": valid_excluded},
    }
    public_payloads = {"manifest": manifest, "train": train, "valid": valid}
    public_data_hash = content_hash(public_payloads)

    # Pass two: independent test cohort. Never intersect it with valid users.
    test: dict[str, dict[str, Any]] = {}
    test_excluded = {"cold_target_outside_train_catalog": 0, "no_held_out_target": 0}
    for user in sorted(users):
        row = users[user]
        target = _integer(row.get("test"), "test target")
        if target == 0:
            test_excluded[_zero_reason(row, "test")] += 1
            continue
        history = train[user] + ([row["valid"]] if row["valid"] > 0 else [])
        if target not in catalog_set:
            raise ValueError("positive test target is outside the training catalog")
        if target in history:
            raise ValueError("repeated test target: register a repeat-aware protocol; do not drop this user")
        test[user] = {"history": history, "target": target}
    if _integer(stats.get("warm_test_users"), "warm_test_users") != len(test):
        raise ValueError("source stats and actual warm test cohort disagree")
    provenance = {
        "adapter_version": ADAPTER_VERSION, "source_protocol": SOURCE_PROTOCOL,
        "prepared_source_sha256": source_sha256, "source_bytes": len(raw),
        "raw_source_sha256": source_files, "source_catalog_sha256": content_hash(catalog_rows),
        "public_data_hash": public_data_hash, "public_content_sha256": public_fingerprint,
        "source_stats": stats, "test_users": len(test), "test_excluded": test_excluded,
        "test_rows_sha256": content_hash(test),
        "test_cohort_sha256": content_hash(sorted(test)),
        "validation_did_not_consult_test": True,
    }
    payloads = {
        "public/manifest.json": manifest, "public/train.json": train,
        "public/valid.json": valid, "private/test.json": test,
        "private/provenance.json": provenance,
    }
    # The exclusive reservation means retries cannot overwrite a complete or partial import.
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    destination.joinpath("private").mkdir(mode=0o700)
    for relative, value in payloads.items():
        atomic_write(confined_path(destination, relative), canonical_json(value).encode("utf-8"))
    bundle = DatasetBundle(destination)
    if bundle.data_hash != public_data_hash:
        raise RuntimeError("imported public bundle identity changed unexpectedly")
    return bundle


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-valid-users", type=int)
    parser.add_argument("--expected-catalog-items", type=int)
    args = parser.parse_args()
    bundle = import_beauty_bundle(args.source, args.destination,
                                 expected_sha256=args.expected_sha256,
                                 expected_valid_users=args.expected_valid_users,
                                 expected_catalog_items=args.expected_catalog_items)
    print(canonical_json(bundle.info()))


if __name__ == "__main__":
    main()
