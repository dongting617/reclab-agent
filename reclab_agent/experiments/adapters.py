"""Trusted dataset and evaluator registration, independent of task launching."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable
from reclab_agent.domain.contracts import canonical_json, identifier, validate_digest
from reclab_agent.data.datasets import DatasetBundle
from reclab_agent.infrastructure.files import confined_path
from reclab_agent.evaluation.ranking import RankingProtocol
from reclab_agent.domain.configuration import read_json_strict


@dataclass(frozen=True)
class EvaluatorDescription:
    """Public metadata for a trusted scorer, not executable code from JSON."""

    protocol_hash: str
    metrics: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        validate_digest(self.protocol_hash, "evaluator protocol_hash")
        if (not isinstance(self.metrics, tuple) or not self.metrics
                or any(not isinstance(metric, str) or not metric.strip() for metric in self.metrics)
                or len(set(self.metrics)) != len(self.metrics)):
            raise ValueError("evaluator metrics must be a nonempty unique tuple of names")
        if not isinstance(self.details, dict):
            raise ValueError("evaluator details must be a JSON object")
        canonical_json(self.details)


def _load_next_item(directory: Path) -> DatasetBundle:
    # Reject ambiguous JSON before DatasetBundle validates the data protocol.
    public = confined_path(directory, "public")
    for name in ("manifest.json", "train.json", "valid.json"):
        read_json_strict(confined_path(public, name))
    bundle = DatasetBundle(directory)
    catalog = bundle.manifest["catalog"]
    if catalog != list(range(1, len(catalog) + 1)):
        raise ValueError("next_item_v1 requires an ordered contiguous catalog from 1 to N")
    return bundle


def _describe_next_item(bundle: DatasetBundle) -> EvaluatorDescription:
    protocol = RankingProtocol(dataset_id=f"{bundle.dataset_id}-{bundle.data_hash}",
                               split="valid", catalog=tuple(bundle.manifest["catalog"]))
    return EvaluatorDescription(protocol.protocol_hash,
                                ("recall@10", "ndcg@10", "recall@20", "ndcg@20"),
                                {"split": "valid", "ranking": "full_catalog_single_target",
                                 "catalog_items": len(protocol.catalog), "ks": list(protocol.ks),
                                 "mask_seen": protocol.mask_seen})


def _score_next_item(bundle: DatasetBundle, predictions: dict[str, Any]) -> dict[str, Any]:
    """Independent validation-set evaluation; test access is not exposed here."""
    return bundle.grade(predictions, "valid")


class TaskAdapterRegistry:
    """Named controller callbacks, never import paths supplied by a task file.

    A dataset loader receives its root Path and returns a DatasetBundle-like
    object exposing public, dataset_id, data_hash and info(); public must be the
    existing configured root/public directory, with no conversion or writes. An evaluator
    description callback receives that object and returns EvaluatorDescription.
    An optional score callback receives (dataset, predictions) and returns an
    independent report. Both callbacks are trusted controller code, never code
    imported from the task configuration or candidate source.
    """

    def __init__(self, *, include_defaults: bool = True):
        self._datasets: dict[str, Callable[[Path], Any]] = {}
        self._evaluators: dict[str, Callable[[Any], EvaluatorDescription]] = {}
        self._scorers: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {}
        if include_defaults:
            self.register_dataset("next_item_v1", _load_next_item)
            self.register_evaluator("next_item_ranking_v1", _describe_next_item, score=_score_next_item)

    @staticmethod
    def _register(registry, name, callback):
        identifier(name)
        if name in registry:
            raise ValueError(f"adapter is already registered: {name}")
        if not callable(callback):
            raise TypeError("adapter callback must be callable trusted controller code")
        registry[name] = callback

    def register_dataset(self, name: str, callback: Callable[[Path], Any]) -> None:
        self._register(self._datasets, name, callback)

    def register_evaluator(self, name: str, callback: Callable[[Any], EvaluatorDescription], *,
                           score: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None) -> None:
        if score is not None and not callable(score):
            raise TypeError("score callback must be callable trusted controller code")
        self._register(self._evaluators, name, callback)
        if score is not None:
            self._scorers[name] = score

    def can_score(self, name: str) -> bool:
        """Return whether this registered evaluator has an executable scorer."""
        return name in self._evaluators and callable(self._scorers.get(name))

    def load_dataset(self, name: str, directory: Path) -> Any:
        if name not in self._datasets:
            raise ValueError(f"dataset adapter is not registered: {name}")
        return self._datasets[name](directory)

    def describe_evaluator(self, name: str, dataset: Any) -> EvaluatorDescription:
        if name not in self._evaluators:
            raise ValueError(f"evaluator is not registered: {name}")
        description = self._evaluators[name](dataset)
        if not isinstance(description, EvaluatorDescription):
            raise TypeError("evaluator callback must return EvaluatorDescription")
        return description

    def score_predictions(self, name: str, dataset: Any,
                          predictions: dict[str, Any]) -> dict[str, Any]:
        """Run a fixed host scorer and bind its report to the declared protocol.

        Prediction content is validated by the task's scorer. The common report
        contract requires matching protocol and evaluation-context identities,
        all declared metrics, and a coherent evaluated-user count. Candidate
        metrics are never accepted as a replacement for this callback.
        """
        description = self.describe_evaluator(name, dataset)
        if not self.can_score(name):
            raise ValueError(f"evaluator has no scoring callback configured: {name}")
        if (not isinstance(predictions, dict) or not predictions
                or any(not isinstance(user, str) or not user.strip() for user in predictions)):
            raise ValueError("predictions must be a nonempty mapping keyed by user IDs")
        report = self._scorers[name](dataset, predictions)
        if not isinstance(report, dict):
            raise ValueError("score callback must return a JSON report object")
        try:
            canonical_json(report)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("score report must contain finite JSON values only") from error
        if (("status" in report and report["status"] != "completed")
                or ("ok" in report and report["ok"] is not True)
                or report.get("failure") is not None or report.get("error") is not None):
            raise ValueError("failed scoring cannot be accepted as a completed score report")
        protocol_hash = validate_digest(report.get("protocol_hash"), "score protocol_hash")
        if protocol_hash != description.protocol_hash:
            raise ValueError("score report protocol_hash differs from its registered evaluator")
        validate_digest(report.get("evaluation_hash"), "score evaluation_hash")
        metrics = report.get("metrics")
        if not isinstance(metrics, dict) or not set(description.metrics) <= metrics.keys():
            raise ValueError("score report must contain every declared metric")
        users = metrics.get("users")
        if type(users) is not int or users <= 0 or users != len(predictions):
            raise ValueError("score report users must match the prediction-user count")
        for metric, value in metrics.items():
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"score metric must be a finite number: {metric}")
        return report
