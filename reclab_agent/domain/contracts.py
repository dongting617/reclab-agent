"""Versioned, provider-neutral contracts for recommendation research tasks."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("identifier must contain 1-96 ASCII letters, digits, dots, underscores or hyphens")
    return value


def validate_digest(value: Any, label: str) -> str:
    """Validate a content or protocol identity shared by host adapters."""
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class Budget:
    max_model_turns: int = 32
    max_patches: int = 6
    max_training_jobs: int = 3
    max_job_seconds: int = 600
    max_total_job_seconds: int = 1800
    max_cost_cny: float = 20.0

    def __post_init__(self) -> None:
        for name in ("max_model_turns", "max_patches", "max_training_jobs", "max_job_seconds", "max_total_job_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.max_cost_cny, bool) or not isinstance(self.max_cost_cny, (int, float)) or not math.isfinite(self.max_cost_cny) or self.max_cost_cny <= 0:
            raise ValueError("max_cost_cny must be finite and positive")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    question: str
    dataset_id: str
    baseline_id: str
    budget: Budget = field(default_factory=Budget)
    mode: str = "single"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value in (self.task_id, self.dataset_id, self.baseline_id):
            identifier(value)
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question cannot be empty")
        if self.mode not in {"single", "plan_execute", "reviewer", "research_audit"}:
            raise ValueError("unsupported agent mode")
        if self.schema_version != 1:
            raise ValueError("unsupported task schema_version")
        if not isinstance(self.budget, Budget):
            raise ValueError("budget must be a Budget")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        payload = dict(value)
        payload["budget"] = Budget(**payload.get("budget", {}))
        return cls(**payload)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    sha256: str
    bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
