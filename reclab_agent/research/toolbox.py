"""Reviewed source edits and development experiments, independent of Agent style.

The injected experiment runner is trusted host configuration. It must enforce
Docker isolation, train/predict/independent scoring, and actual time/cost limits.
This toolbox bounds attempts within one live session; it is not a durable job
coordinator and does not execute candidate Python on the host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..contracts import TaskSpec
from ..tools.contracts import ToolError, ToolSpec
from ..tools.registry import ToolRegistry
from ..workspace import Workspace
from .contracts import ExperimentRecord, PlanReview, ResearchPlan
from .workflow import require_plan_approval


class ResearchToolbox:
    """Expose only the capabilities explicitly configured by the host."""

    def __init__(self, workspace: Workspace, *, task: TaskSpec, plan: ResearchPlan,
                 review: PlanReview | None, snapshots_root: str | Path,
                 run_experiment: Callable[..., ExperimentRecord] | None = None):
        self.workspace, self.task, self.plan, self.review = workspace, task, plan, review
        self.snapshots_root = Path(snapshots_root)
        self._runner = run_experiment
        self._patches = self._attempts = self._reserved_seconds = 0
        self._candidates: set[str] = set()
        self._records: dict[str, ExperimentRecord] = {}
        self._completed_requests: dict[tuple[str, str, int], str] = {}
        self._validate_scope()
        if run_experiment is not None and not callable(run_experiment):
            raise TypeError("run_experiment must be a trusted callable")
        self._tools = ToolRegistry()
        self._register("read_source", "Read an approved candidate file and its current hash.",
                       {"path": {"type": "string"}}, self.read_source)
        self._register("apply_patch", "Apply one exact, hash-checked edit within the reviewed plan.", {
            "path": {"type": "string"}, "old": {"type": "string"},
            "new": {"type": "string"}, "expected_sha256": {"type": "string"},
        }, self.apply_patch)
        self._register("validate_candidate", "Check candidate syntax without executing it.", {},
                       self.validate_candidate)
        if self._runner is not None:
            self._register("run_experiment", "Freeze and run one reviewed development experiment through the trusted runner.",
                           {"candidate_id": {"type": "string"}, "seed": {"type": "integer"}},
                           self.run_experiment)
            self._register("get_experiment", "Read a recorded experiment and its independent score reference.",
                           {"experiment_id": {"type": "string"}}, self.get_experiment)

    def _register(self, name: str, description: str, properties: dict[str, Any], handler: Callable) -> None:
        self._tools.register_function(ToolSpec(name, description, {
            "type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False,
        }), handler)

    @property
    def specs(self) -> list[ToolSpec]:
        return self._tools.specs

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._tools.invoke(name, arguments)

    @property
    def usage(self) -> dict[str, Any]:
        return {"patch_attempts": self._patches, "experiment_attempts": self._attempts,
                "candidate_ids": sorted(self._candidates),
                "reserved_training_seconds": self._reserved_seconds,
                "actual_time_and_cost_enforcement": "trusted_runner"}

    @property
    def experiments(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records.values()]

    def _validate_scope(self) -> None:
        self.plan.validate_for_task(self.task)
        if self.workspace.editable_paths != frozenset(self.plan.editable_files):
            raise ValueError("workspace files must exactly match the reviewed plan")

    def _authorize(self) -> None:
        self._validate_scope()
        try:
            require_plan_approval(self.plan, self.review)
        except ValueError as error:
            raise ToolError("PLAN_NOT_APPROVED", str(error)) from error

    def read_source(self, path: str) -> dict[str, Any]:
        return self.workspace.read(path)

    def apply_patch(self, path: str, old: str, new: str, expected_sha256: str) -> dict[str, Any]:
        self._authorize()
        if path not in self.plan.editable_files:
            raise ToolError("OUTSIDE_PLAN", "file is outside the reviewed plan")
        if self._patches >= self.plan.budget.max_patches:
            raise ToolError("BUDGET_EXCEEDED", "patch attempt budget exhausted")
        self._patches += 1
        return self.workspace.replace(path, old, new, expected_sha256)

    def validate_candidate(self) -> dict[str, Any]:
        return self.workspace.validate()

    def run_experiment(self, candidate_id: str, seed: int) -> dict[str, Any]:
        self._authorize()
        if self._runner is None:
            raise ToolError("RUNNER_NOT_CONFIGURED", "no trusted experiment runner is configured")
        try:
            self.plan.check_scope(candidate_id, tuple(sorted(self.workspace.editable_paths)), seed)
        except ValueError as error:
            raise ToolError("OUTSIDE_PLAN", str(error)) from error
        validation = self.workspace.validate()
        if not validation["ok"]:
            raise ToolError("CANDIDATE_INVALID", str(validation["errors"]))
        snapshot = self.workspace.snapshot(self.snapshots_root)
        request = (snapshot["source_hash"], candidate_id, seed)
        if request in self._completed_requests:
            return self.get_experiment(self._completed_requests[request])
        remaining = self.plan.budget.max_total_job_seconds - self._reserved_seconds
        if self._attempts >= self.plan.budget.max_training_jobs or remaining <= 0:
            raise ToolError("BUDGET_EXCEEDED", "experiment attempt or reserved time budget exhausted")
        if candidate_id not in self._candidates and len(self._candidates) >= self.plan.max_candidates:
            raise ToolError("BUDGET_EXCEEDED", "candidate budget exhausted")
        max_seconds = min(self.plan.budget.max_job_seconds, remaining)
        # Charge before crossing the execution boundary, including runner errors.
        self._attempts += 1
        self._reserved_seconds += max_seconds
        self._candidates.add(candidate_id)
        expected = {"plan_hash": self.plan.plan_hash, "source_hash": snapshot["source_hash"],
                    "candidate_id": candidate_id, "seed": seed, "dataset_id": self.plan.dataset_id,
                    "protocol_hash": self.plan.protocol_hash}
        try:
            record = self._runner(snapshot, candidate_id=candidate_id, seed=seed, max_seconds=max_seconds)
        except Exception as error:
            raise ToolError("EXPERIMENT_RUN_FAILED", f"{type(error).__name__}: {error}") from error
        if (not isinstance(record, ExperimentRecord)
                or any(getattr(record, field) != value for field, value in expected.items())
                or record.status in {"planned", "running"}):
            raise ToolError("INVALID_EXPERIMENT_RESULT", "runner must return a matching terminal ExperimentRecord")
        if record.experiment_id in self._records:
            raise ToolError("INVALID_EXPERIMENT_RESULT", "experiment ID was already used for another request")
        self._records[record.experiment_id] = record
        self._completed_requests[request] = record.experiment_id
        return record.to_dict()

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        try:
            return self._records[experiment_id].to_dict()
        except KeyError as error:
            raise ToolError("UNKNOWN_EXPERIMENT", "experiment is not recorded in this session") from error
