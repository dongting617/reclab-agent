"""Experiments function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.domain.research import ExperimentRecord
from reclab_agent.tools.contracts import ToolError
from reclab_agent.tools.tool_group import ToolGroup

class ExperimentTools(ToolGroup):
    definitions = (('run_experiment',
      'Freeze one candidate and automatically train, predict, and independently score it '
      'within the reviewed budget.',
      {'candidate_id': {'type': 'string'}, 'seed': {'type': 'integer'}}),
     ('get_experiment',
      'Read a recorded experiment and its independent score reference.',
      {'experiment_id': {'type': 'string'}}),
     ('read_log',
      'Read a recorded experiment log, not model-reported metrics.',
      {'path': {'type': 'string'}}),
     ('list_experiments',
      "Read the durable experimental ledger for this project's comparison scope.",
      {}))

    def run_experiment(self, candidate_id: str, seed: int) -> dict[str, Any]:
        self._authorize()
        if self.active_stage is not None:
            if self.active_stage == "background":
                if candidate_id != self.task.baseline_id:
                    raise ToolError("WRONG_STAGE", "background may only establish the baseline")
            elif (self.active_stage != "experiment" or not self.hypothesis or candidate_id == self.task.baseline_id
                  or (self.active_candidate_id is not None and candidate_id != self.active_candidate_id)):
                raise ToolError("WRONG_STAGE", "candidate execution requires a registered hypothesis")
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
        if self.active_stage == "background" and snapshot["source_hash"] != self.baseline_source_hash:
            raise ToolError("BASELINE_MODIFIED", "baseline source must match the registered snapshot")
        request = (snapshot["source_hash"], candidate_id, seed)
        if request in self._completed_requests:
            return self.get_experiment(self._completed_requests[request])
        remaining = self.plan.budget.max_total_job_seconds - self._reserved_seconds
        recovering = (self.resuming and self.pending_request is not None
                      and request == tuple(self.pending_request["request"]))
        if not recovering and (self._attempts >= self.plan.budget.max_training_jobs or remaining <= 0):
            raise ToolError("BUDGET_EXCEEDED", "experiment attempt or reserved time budget exhausted")
        if candidate_id not in self._candidates and len(self._candidates) >= self.plan.max_candidates:
            raise ToolError("BUDGET_EXCEEDED", "candidate budget exhausted")
        max_seconds = self.pending_request["seconds"] if recovering else min(self.plan.budget.max_job_seconds, remaining)
        if self.deadline is not None and not recovering:
            import time
            max_seconds = min(max_seconds, int(self.deadline - time.time()))
        if max_seconds < 2:
            raise ToolError("BUDGET_EXCEEDED", "insufficient time for train and predict")
        # Charge before crossing the execution boundary, including runner errors.
        if not recovering:
            self._attempts += 1
            self._reserved_seconds += max_seconds
        self._candidates.add(candidate_id)
        self.pending_request = {"request": list(request), "seconds": max_seconds}
        self.save_state()
        commit = self.project.commit_source(snapshot, snapshot["source_hash"]) if self.project else None
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
        if self.state_path is not None and (self.state_path.parent / "active-remote-job.json").exists():
            raise ToolError("RECONCILIATION_REQUIRED", "remote job is still recorded as active; preserve the pending request and reconcile it before recording a terminal result")
        if record.experiment_id in self._records:
            raise ToolError("INVALID_EXPERIMENT_RESULT", "experiment ID was already used for another request")
        self._records[record.experiment_id] = record
        self._completed_requests[request] = record.experiment_id
        self.git_commits[record.experiment_id] = commit
        self.pending_request = None
        self.save_state()
        return record.to_dict()


    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        try:
            return self._records[experiment_id].to_dict()
        except KeyError as error:
            raise ToolError("UNKNOWN_EXPERIMENT", "experiment is not recorded in this session") from error


    def read_log(self, path: str) -> dict[str, Any]:
        from reclab_agent.infrastructure.files import confined_path
        from reclab_agent.infrastructure.files import read_regular_file
        if not path.startswith("experiments/") or not path.endswith((".json", ".jsonl", ".log")):
            raise ValueError("only experiment logs are readable")
        raw = read_regular_file(confined_path(self.snapshots_root.parent, path), max_bytes=2_000_000)
        return {"path": path, "text": raw.decode("utf-8", errors="replace")}


    def list_experiments(self) -> dict[str, Any]:
        return {"experiments": self.project.experiments(self.scope)}
