"""Shared study state, scope authorization, budgets and checkpoint recovery.

Capability groups in tools/ own callable handlers. The trusted experiment runner
enforces execution limits; this context persists counters and recorded results.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
from reclab_agent.domain.contracts import TaskSpec
from reclab_agent.tools.contracts import ToolError
from reclab_agent.experiments.workspace import Workspace
from reclab_agent.domain.research import ExperimentRecord, PlanReview, ResearchPlan
from reclab_agent.domain.authorization import require_plan_approval


class StudyContext:
    """Shared study state, authorization, budgets and checkpoint persistence."""

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
        self.active_stage = None
        self.hypothesis = None
        self.baseline_source_hash = None
        self.active_candidate_id = None
        self.project = None
        self.scope = None
        self.state_path = None
        self.pending_request = None
        self.deadline = None
        self.resuming = False
        self.git_commits = {}
        self.operations = {}
        self.literature = None
        self._validate_scope()
        if run_experiment is not None and not callable(run_experiment):
            raise TypeError("run_experiment must be a trusted callable")
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
        self.check_deadline()
        self._validate_scope()
        try:
            require_plan_approval(self.plan, self.review)
        except ValueError as error:
            raise ToolError("PLAN_NOT_APPROVED", str(error)) from error


    def restore_source(self, source_hash: str) -> dict[str, Any]:
        from reclab_agent.experiments.outcomes import restore_snapshot
        return restore_snapshot(self, source_hash)

    def check_deadline(self):
        import time
        if self.deadline is not None and time.time() >= self.deadline:
            raise ToolError("BUDGET_EXCEEDED", "task deadline exceeded")

    def attach(self, project, scope, output, deadline, *, resume=False):
        from reclab_agent.domain.configuration import read_json_strict
        self.project, self.scope, self.deadline = project, scope, deadline
        self.state_path = Path(output) / "service-state.json"
        self.resuming = resume
        if resume and self.state_path.exists():
            state = read_json_strict(self.state_path, max_bytes=8_000_000)
            if state["plan_hash"] != self.plan.plan_hash:
                raise ValueError("cannot resume under a different task scope")
            self._patches, self._attempts, self._reserved_seconds = state["counters"]
            self._candidates = set(state["candidates"])
            self._records = {v["experiment_id"]: ExperimentRecord.from_dict(v) for v in state["records"]}
            self._completed_requests = {tuple(row[:3]): row[3] for row in state["completed"]}
            self.pending_request = state["pending"]
            self.hypothesis = state["hypothesis"]
            self.git_commits = state.get("git_commits", {})
            self.workspace.source_paths = frozenset(state.get("source_paths", self.workspace.source_paths))
            self.restore_source(state["source_hash"])
        self.save_state()

    def save_state(self):
        from reclab_agent.domain.contracts import canonical_json
        from reclab_agent.infrastructure.files import atomic_write
        if self.state_path is None:
            return
        snapshot = self.workspace.snapshot(self.snapshots_root)
        state = {"plan_hash": self.plan.plan_hash, "source_hash": snapshot["source_hash"],
                 "counters": [self._patches, self._attempts, self._reserved_seconds],
                 "candidates": sorted(self._candidates), "records": self.experiments,
                 "completed": [[*key, value] for key, value in self._completed_requests.items()],
                 "pending": self.pending_request, "hypothesis": self.hypothesis,
                 "stage": self.active_stage, "git_commits": self.git_commits,
                 "source_paths": sorted(self.workspace.source_paths)}
        atomic_write(self.state_path, canonical_json(state).encode("utf-8"))
        if self.project:
            self.project.save_run(self.state_path.parent.name, state, "research_state")
