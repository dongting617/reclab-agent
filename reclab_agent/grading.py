"""Independent, deterministic full-catalog ranking metrics.

The caller supplies trusted targets and histories. This module only accepts
plain data: it never imports candidate code or loads model checkpoints.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _item(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer (not bool)")
    return value


def _users(value: Any, label: str) -> set[str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict keyed by user ID")
    if any(not isinstance(key, str) or not key.strip() for key in value):
        raise ValueError(f"{label} user IDs must be nonempty strings")
    return set(value)


@dataclass(frozen=True)
class RankingProtocol:
    """A frozen full-catalog, one-target-per-user evaluation protocol."""

    dataset_id: str
    split: str
    catalog: tuple[int, ...]
    ks: tuple[int, ...] = (10, 20)
    mask_seen: bool = True

    def __post_init__(self) -> None:
        for name in ("dataset_id", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.mask_seen) is not bool:
            raise ValueError("mask_seen must be a bool")
        for name in ("catalog", "ks"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            for value in values:
                _item(value, name)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique integers")
            object.__setattr__(self, name, tuple(sorted(values)))
        if any(k <= 0 for k in self.ks):
            raise ValueError("ks must contain positive integers")

    def to_dict(self) -> dict[str, Any]:
        return {"dataset_id": self.dataset_id, "split": self.split,
                "catalog": list(self.catalog), "ks": list(self.ks),
                "mask_seen": self.mask_seen, "ranking": "full_catalog_single_target"}

    @property
    def protocol_hash(self) -> str:
        return _hash(self.to_dict())


def grade_rankings(
    predictions: dict[str, list[int]],
    targets: dict[str, int],
    histories: dict[str, list[int]],
    protocol: RankingProtocol,
) -> dict[str, Any]:
    """Grade ordered top-K item IDs against externally supplied targets.

    Predictions must cover every target user exactly once and provide at least
    min(max(K), eligible catalog size) unique eligible items. Additional ranked
    items are permitted, but are validated just as strictly. An absent target
    receives zero metrics, never an inferred score from candidate output.
    """
    if not isinstance(protocol, RankingProtocol):
        raise ValueError("protocol must be a RankingProtocol")
    users = _users(targets, "targets")
    if not users:
        raise ValueError("targets must contain at least one user")
    if _users(predictions, "predictions") != users:
        raise ValueError("predictions must cover exactly the target users")
    if _users(histories, "histories") != users:
        raise ValueError("histories must cover exactly the target users")

    catalog = set(protocol.catalog)
    per_user: dict[str, dict[str, Any]] = {}
    context: dict[str, Any] = {}
    metric_names = [f"{name}@{k}" for k in protocol.ks for name in ("recall", "ndcg")]
    for user_id in sorted(users):
        target = _item(targets[user_id], f"target for {user_id}")
        if target not in catalog:
            raise ValueError(f"target for {user_id} is outside the catalog")
        history = histories[user_id]
        prediction = predictions[user_id]
        if not isinstance(history, list) or not isinstance(prediction, list):
            raise ValueError("histories and predictions must contain lists")
        for value in history:
            if _item(value, f"history for {user_id}") not in catalog:
                raise ValueError(f"history for {user_id} contains an unknown item")
        seen = set(history)
        if protocol.mask_seen and target in seen:
            raise ValueError(f"target for {user_id} is masked by its history")
        eligible = catalog - seen if protocol.mask_seen else catalog
        for value in prediction:
            if _item(value, f"prediction for {user_id}") not in eligible:
                raise ValueError(f"prediction for {user_id} contains an ineligible item")
        if len(set(prediction)) != len(prediction):
            raise ValueError(f"prediction for {user_id} contains duplicate items")
        required = min(max(protocol.ks), len(eligible))
        if len(prediction) < required:
            raise ValueError(f"prediction for {user_id} needs at least {required} items")

        rank = prediction.index(target) + 1 if target in prediction else None
        row: dict[str, Any] = {"rank": rank}
        for k in protocol.ks:
            hit = rank is not None and rank <= k
            row[f"recall@{k}"] = float(hit)
            row[f"ndcg@{k}"] = 1.0 / math.log2(rank + 1) if hit else 0.0
        per_user[user_id] = row
        # Repeated interactions and their order do not change seen-item masking.
        context[user_id] = {"target": target, "seen": sorted(seen)}

    metrics: dict[str, Any] = {"users": len(users)}
    for name in metric_names:
        metrics[name] = math.fsum(row[name] for row in per_user.values()) / len(users)
    return {"schema_version": 1, "protocol": protocol.to_dict(),
            "protocol_hash": protocol.protocol_hash,
            "evaluation_hash": _hash({"protocol_hash": protocol.protocol_hash, "users": context}),
            "metrics": metrics, "per_user": per_user}


def _report_values(report: Any) -> tuple[list[str], dict[str, list[float]]]:
    if not isinstance(report, dict):
        raise ValueError("reports must be dicts")
    try:
        rows, metrics = report["per_user"], report["metrics"]
        users = sorted(_users(rows, "per_user"))
        protocol = report["protocol"]
        if not isinstance(protocol, dict) or _hash(protocol) != report["protocol_hash"]:
            raise ValueError("report protocol hash does not match its protocol")
        if not users or type(metrics["users"]) is not int or metrics["users"] != len(users):
            raise ValueError("report user count is inconsistent")
        if not isinstance(report["evaluation_hash"], str) or not report["evaluation_hash"]:
            raise ValueError("report is missing its evaluation identity")
        names = sorted(key for key in metrics if key != "users")
        expected = sorted(f"{name}@{k}" for k in protocol["ks"] for name in ("recall", "ndcg"))
        if names != expected:
            raise ValueError("report metric names do not match its protocol")
        values: dict[str, list[float]] = {}
        for name in names:
            series = [rows[user][name] for user in users]
            for value in [metrics[name], *series]:
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("report metrics must be finite numbers in [0, 1]")
            mean = math.fsum(series) / len(users)
            if not math.isclose(metrics[name], mean, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("report aggregate does not match per-user metrics")
            values[name] = series
        return users, values
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid ranking report") from exc


def _quantile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return candidate-minus-baseline deltas and paired 95% bootstrap CIs.

    Users are aligned by ID. Each bootstrap replicate resamples the same users
    for all metrics, retaining within-user pairing. Intervals describe sampled
    users, not variation across training seeds or repeated experiment selection.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    _item(seed, "seed")
    users, before = _report_values(baseline)
    candidate_users, after = _report_values(candidate)
    if baseline["protocol_hash"] != candidate["protocol_hash"]:
        raise ValueError("cannot compare reports from different protocols or splits")
    if users != candidate_users:
        raise ValueError("cannot compare reports with different users")
    if baseline["evaluation_hash"] != candidate["evaluation_hash"]:
        raise ValueError("cannot compare reports with different targets or histories")
    if before.keys() != after.keys():
        raise ValueError("cannot compare different metric sets")

    deltas = {name: [new - old for old, new in zip(before[name], after[name])]
              for name in before}
    samples: dict[str, list[float]] = {name: [] for name in before}
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(users)) for _ in users]
        for name, values in deltas.items():
            samples[name].append(math.fsum(values[index] for index in indices) / len(users))
    comparisons: dict[str, Any] = {}
    for name, values in deltas.items():
        ordered = sorted(samples[name])
        comparisons[name] = {
            "baseline": baseline["metrics"][name],
            "candidate": candidate["metrics"][name],
            "mean_delta": math.fsum(values) / len(users),
            "ci95": [_quantile(ordered, 0.025), _quantile(ordered, 0.975)],
        }
    return {"protocol_hash": baseline["protocol_hash"],
            "evaluation_hash": baseline["evaluation_hash"], "users": len(users),
            "bootstrap_samples": bootstrap_samples, "seed": seed,
            "interval": "paired_percentile_95", "metrics": comparisons}
