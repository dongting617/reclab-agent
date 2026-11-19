"""JSON contracts for reviewed research; no execution or implicit approval.

Evidence references identify supporting material, not proof that it is correct.
Only a trusted coordinator may record a user's review or verify actual artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any

from ..contracts import Budget, TaskSpec, content_hash, identifier


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _tuple(value: tuple, name: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, tuple) or (nonempty and not value):
        raise ValueError(f"{name} must be {'a nonempty' if nonempty else 'a'} tuple")


def _texts(value: tuple[str, ...], name: str, *, nonempty: bool = False) -> None:
    _tuple(value, name, nonempty=nonempty)
    for item in value:
        _text(item, name)


def _sequence(value: Any, name: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    return tuple(value)


def _source_path(value: str) -> None:
    _text(value, "editable file")
    path, windows = PurePosixPath(value), PureWindowsPath(value)
    if ("\\" in value or "\x00" in value or path.is_absolute() or windows.drive
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or not value.startswith(("src/", "configs/"))
            or path.suffix not in {".py", ".json", ".yaml", ".yml"}):
        raise ValueError("editable files must be relative src/configs text paths")


class _JsonContract:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRef(_JsonContract):
    source: str
    sha256: str
    location: str
    conditions: str

    def __post_init__(self) -> None:
        for name in ("source", "location", "conditions"):
            _text(getattr(self, name), name)
        _digest(self.sha256, "sha256")


@dataclass(frozen=True)
class ResearchClaim(_JsonContract):
    statement: str
    kind: str
    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        _text(self.statement, "statement")
        if self.kind not in {"observation", "inference", "hypothesis"}:
            raise ValueError("claim kind must distinguish observation, inference and hypothesis")
        _tuple(self.evidence, "evidence", nonempty=self.kind == "observation")
        if any(not isinstance(ref, EvidenceRef) for ref in self.evidence):
            raise ValueError("evidence must contain EvidenceRef objects")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchClaim:
        payload = dict(value)
        payload["evidence"] = tuple(EvidenceRef(**ref) for ref in _sequence(payload.get("evidence", ()), "evidence"))
        return cls(**payload)


@dataclass(frozen=True)
class ResearchPlan(_JsonContract):
    plan_id: str
    task_id: str
    hypothesis: str
    dataset_id: str
    data_profile: str
    baseline_id: str
    protocol_hash: str
    primary_metric: str
    editable_files: tuple[str, ...]
    comparison: str
    expected_observation: str
    rejection_criteria: str
    stop_conditions: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...] = ()
    guardrail_metrics: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ("candidate-1",)
    seeds: tuple[int, ...] = (42,)
    budget: Budget = field(default_factory=Budget)
    max_candidates: int = 2
    confirmation_included: bool = False
    revision: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("plan_id", "task_id", "dataset_id", "baseline_id"):
            identifier(getattr(self, name))
        for name in ("hypothesis", "data_profile", "primary_metric", "comparison",
                     "expected_observation", "rejection_criteria"):
            _text(getattr(self, name), name)
        _digest(self.protocol_hash, "protocol_hash")
        _texts(self.editable_files, "editable_files", nonempty=True)
        for path in self.editable_files:
            _source_path(path)
        _texts(self.stop_conditions, "stop_conditions", nonempty=True)
        _texts(self.guardrail_metrics, "guardrail_metrics")
        _texts(self.candidate_ids, "candidate_ids", nonempty=True)
        for candidate in self.candidate_ids:
            identifier(candidate)
        if (type(self.max_candidates) is not int or self.max_candidates < 1
                or len(self.candidate_ids) > self.max_candidates):
            raise ValueError("candidate count exceeds the plan limit")
        for name in ("editable_files", "candidate_ids", "seeds"):
            values = getattr(self, name)
            _tuple(values, name, nonempty=True)
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique values")
        if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative 31-bit integers")
        _tuple(self.evidence, "evidence")
        if any(not isinstance(ref, EvidenceRef) for ref in self.evidence):
            raise ValueError("evidence must contain EvidenceRef objects")
        if not isinstance(self.budget, Budget) or type(self.confirmation_included) is not bool:
            raise ValueError("invalid budget or confirmation flag")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported research plan schema_version")

    @property
    def plan_hash(self) -> str:
        """Bind review to the complete plan, including scope and resource limits."""
        return content_hash(self.to_dict())

    def validate_for_task(self, task: TaskSpec) -> None:
        if (self.task_id, self.dataset_id, self.baseline_id) != (task.task_id, task.dataset_id, task.baseline_id):
            raise ValueError("plan does not match the task identity")
        if any(value > getattr(task.budget, name) for name, value in asdict(self.budget).items()):
            raise ValueError("plan budget exceeds the task budget")

    def check_scope(self, candidate_id: str, paths: tuple[str, ...], seed: int, *, confirmation: bool = False) -> None:
        """Check approved scope only; actual budget accounting belongs to the coordinator."""
        _texts(paths, "paths")
        if (candidate_id not in self.candidate_ids or type(seed) is not int or seed not in self.seeds
                or not set(paths) <= set(self.editable_files)
                or type(confirmation) is not bool or (confirmation and not self.confirmation_included)):
            raise ValueError("action is outside the reviewed plan scope")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchPlan:
        payload = dict(value)
        for name in ("editable_files", "stop_conditions", "guardrail_metrics", "candidate_ids", "seeds"):
            if name in payload:
                payload[name] = _sequence(payload[name], name)
        payload["budget"] = Budget(**payload.get("budget", {}))
        payload["evidence"] = tuple(EvidenceRef(**ref) for ref in _sequence(payload.get("evidence", ()), "evidence"))
        return cls(**payload)


@dataclass(frozen=True)
class PlanReview(_JsonContract):
    """Trusted user review record, never an LLM self-authorization token."""
    plan_hash: str
    decision: str
    reviewer: str
    reviewed_at: str
    note: str = ""

    def __post_init__(self) -> None:
        _digest(self.plan_hash, "plan_hash")
        if self.decision not in {"approved", "rejected"}:
            raise ValueError("unsupported review decision")
        _text(self.reviewer, "reviewer")
        _text(self.reviewed_at, "reviewed_at")
        if not isinstance(self.note, str):
            raise ValueError("note must be text")


@dataclass(frozen=True)
class ExperimentRecord(_JsonContract):
    experiment_id: str
    plan_hash: str
    candidate_id: str
    source_hash: str
    config_hash: str
    dataset_id: str
    protocol_hash: str
    image: str
    seed: int
    status: str = "planned"
    program_validation: str = "pending"
    training_job_id: str | None = None
    prediction_job_id: str | None = None
    score: EvidenceRef | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        for name in ("experiment_id", "candidate_id", "dataset_id"):
            identifier(getattr(self, name))
        for name in ("plan_hash", "source_hash", "config_hash", "protocol_hash"):
            _digest(getattr(self, name), name)
        if not isinstance(self.image, str) or not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[a-f0-9]{64}", self.image):
            raise ValueError("experiment image must be an immutable repository digest")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be a nonnegative 31-bit integer")
        if self.status not in {"planned", "running", "completed", "failed", "cancelled", "timed_out", "lost"}:
            raise ValueError("unsupported experiment status")
        if self.program_validation not in {"pending", "passed", "failed"}:
            raise ValueError("unsupported program validation status")
        for job_id in (self.training_job_id, self.prediction_job_id):
            if job_id is not None:
                identifier(job_id)
        if self.score is not None and not isinstance(self.score, EvidenceRef):
            raise ValueError("score must reference an independent scoring artifact")
        if self.failure is not None:
            _text(self.failure, "failure")
        if self.status == "completed" and (self.program_validation != "passed" or self.score is None
                                           or self.training_job_id is None or self.prediction_job_id is None):
            raise ValueError("completed experiment requires validated execution, jobs and independent scoring")
        if self.score is not None and self.status != "completed":
            raise ValueError("only completed experiments can carry an independent score")
        if self.status == "completed" and self.failure is not None:
            raise ValueError("execution failure cannot be represented as a completed experiment")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExperimentRecord:
        payload = dict(value)
        if payload.get("score") is not None:
            payload["score"] = EvidenceRef(**payload["score"])
        return cls(**payload)


@dataclass(frozen=True)
class ResearchConclusion(_JsonContract):
    task_id: str
    plan_hash: str
    decision: str
    findings: tuple[ResearchClaim, ...]
    stop_reason: str
    experiment_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    confirmed: bool = False

    def __post_init__(self) -> None:
        identifier(self.task_id)
        _digest(self.plan_hash, "plan_hash")
        if self.decision not in {"retain", "reject", "inconclusive", "investigate"}:
            raise ValueError("unsupported research decision")
        _tuple(self.findings, "findings")
        if any(not isinstance(claim, ResearchClaim) for claim in self.findings):
            raise ValueError("findings must contain ResearchClaim objects")
        _text(self.stop_reason, "stop_reason")
        for name in ("experiment_ids", "limitations", "next_steps"):
            _texts(getattr(self, name), name)
        for experiment_id in self.experiment_ids:
            identifier(experiment_id)
        if type(self.confirmed) is not bool:
            raise ValueError("confirmed must be bool")
        if self.confirmed and (not self.experiment_ids or not any(claim.kind == "observation" for claim in self.findings)):
            raise ValueError("confirmed conclusion requires experiments and referenced observations")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchConclusion:
        payload = dict(value)
        payload["findings"] = tuple(ResearchClaim.from_dict(claim) for claim in _sequence(payload["findings"], "findings"))
        for name in ("experiment_ids", "limitations", "next_steps"):
            if name in payload:
                payload[name] = _sequence(payload[name], name)
        return cls(**payload)
