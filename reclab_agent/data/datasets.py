"""Explicit dataset adapter boundary; test labels never enter candidate mounts."""
from __future__ import annotations
from collections import Counter
from itertools import islice
import json
from pathlib import Path
from typing import Any
from reclab_agent.domain.contracts import content_hash, identifier
from reclab_agent.memory.storage import atomic_write, confined_path


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class DatasetBundle:
    """Public files are mountable. private/ belongs only to the trusted scorer."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.public = confined_path(self.root, "public")
        self.manifest = _read(confined_path(self.public, "manifest.json"))
        identifier(self.manifest["dataset_id"])
        if self.manifest.get("schema_version") != 1 or self.manifest.get("task") != "next_item":
            raise ValueError("unsupported dataset schema/task; register an adapter")
        catalog = self.manifest["catalog"]
        if not isinstance(catalog, list) or not catalog or any(type(i) is not int or i <= 0 for i in catalog) or len(set(catalog)) != len(catalog):
            raise ValueError("catalog must contain unique positive integer item IDs")
        train = _read(confined_path(self.public, "train.json"))
        valid = _read(confined_path(self.public, "valid.json"))
        catalog_set = set(catalog)
        if not isinstance(train, dict) or not train:
            raise ValueError("train must map user IDs to training histories")
        if not isinstance(valid, dict) or not valid:
            raise ValueError("valid must be a nonempty mapping of user IDs to validation rows")
        for user, items in train.items():
            if not isinstance(user, str) or not user or not isinstance(items, list) or any(type(i) is not int or i not in catalog_set for i in items):
                raise ValueError("invalid training history")
        for user, row in valid.items():
            if user not in train or not isinstance(row, dict) or set(row) != {"history", "target"}:
                raise ValueError("invalid validation row")
            if row["history"] != train[user] or type(row["target"]) is not int or row["target"] not in catalog_set or row["target"] in row["history"]:
                raise ValueError("validation history/target violates next-item protocol")
        self.train = train
        self.valid = valid
        self.data_hash = content_hash({"manifest": self.manifest, "train": train, "valid": valid})

    @property
    def dataset_id(self) -> str:
        return self.manifest["dataset_id"]

    def info(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "data_hash": self.data_hash,
                "profile": self.manifest["profile"], "task": "next_item",
                "catalog_items": len(self.manifest["catalog"]), "train_users": len(self.train),
                "valid_users": len(self.valid), "test_labels_exposed": False,
                "quality_evidence": self.manifest.get("quality_evidence", "not_established")}

    def evaluation_inputs(self, split: str) -> dict[str, list[int]]:
        rows = self._rows(split)
        return {user: row["history"] for user, row in rows.items()}

    def _rows(self, split: str) -> dict[str, Any]:
        if split == "valid":
            return self.valid
        if split != "test":
            raise ValueError("split must be valid or test")
        # Only trusted host code calls this; never pass private/ to an executor.
        return _read(confined_path(self.root, "private/test.json"))

    def grade(self, predictions: dict[str, list[int]], split: str = "valid") -> dict[str, Any]:
        from reclab_agent.evaluation.ranking import RankingProtocol
        from reclab_agent.evaluation.ranking import grade_rankings
        rows = self._rows(split)
        protocol = RankingProtocol(dataset_id=f"{self.dataset_id}-{self.data_hash}", split=split,
                                   catalog=tuple(self.manifest["catalog"]))
        return grade_rankings(predictions, {u: r["target"] for u, r in rows.items()},
                              {u: r["history"] for u, r in rows.items()}, protocol)


def create_synthetic_bundle(root: str | Path) -> DatasetBundle:
    """A deterministic executable fixture, explicitly NOT real recommendation evidence."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("dataset destination must be empty")
    catalog = list(range(1, 41))
    train = {f"u{i:02d}": [1 + ((i * 3 + j) % 30) for j in range(5)] for i in range(12)}
    valid = {u: {"history": history, "target": 31 + (i % 5)} for i, (u, history) in enumerate(train.items())}
    test = {u: {"history": row["history"] + [row["target"]], "target": 36 + (i % 5)}
            for i, (u, row) in enumerate(valid.items())}
    manifest = {"schema_version": 1, "dataset_id": "synthetic-smoke-v1", "profile": "synthetic-smoke",
                "task": "next_item", "catalog": catalog, "quality_evidence": "fixture_only",
                "description": "Synthetic contract fixture; do not report these scores as real-world model quality."}
    for relative, value in {"public/manifest.json": manifest, "public/train.json": train,
                            "public/valid.json": valid, "private/test.json": test}.items():
        atomic_write(confined_path(root, relative), json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return DatasetBundle(root)


def popularity_predictions(bundle: DatasetBundle, histories: dict[str, list[int]]) -> dict[str, list[int]]:
    """Trusted deterministic baseline, not a candidate-code execution shortcut."""
    counts = Counter(item for row in bundle.train.values() for item in row)
    catalog = sorted(bundle.manifest["catalog"], key=lambda item: (-counts[item], item))
    predictions = {}
    for user, history in histories.items():
        seen = set(history)
        predictions[user] = list(islice((item for item in catalog if item not in seen), 20))
    return predictions
