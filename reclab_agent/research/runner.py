"""Reusable experiment tool implementation, independent of the target model.

Only the configured Worker executes candidate entrypoints. The controller freezes
inputs, submits train/predict jobs, and sends plain predictions to a fixed scorer.
No candidate import, arbitrary shell, or native execution fallback exists here.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time
from typing import Any, Callable

from ..contracts import canonical_json, content_hash, identifier
from ..execution import DockerWorker, InputTree, JobSpec, ResourceLimits, WorkerPolicy
from ..files import atomic_write, confined_path
from .._execution.locking import job_lock
from .configuration import TaskConfiguration, read_json_strict
from .contracts import EvidenceRef, ExperimentRecord, ResearchPlan
from .inspection import TaskAdapterRegistry


TERMINAL = frozenset({"completed", "failed", "timed_out", "cancelled", "lost"})


class _PhaseFailure(RuntimeError):
    def __init__(self, message: str, status: str = "failed"):
        super().__init__(message)
        self.status = status


class ExperimentRunner:
    """One generic train -> predict -> independent-score pipeline.

    The approved policy is an execution template bound to the original source
    and public dataset. Reviewed edits may replace only the configured editable
    files. Each experiment has a stable Worker directory; an OS lock serializes
    calls sharing this task's output root. This is not a global GPU scheduler.
    A polling deadline supplements Worker limits, not an external hard supervisor.
    """

    def __init__(self, configuration: TaskConfiguration, plan: ResearchPlan,
                 policy: WorkerPolicy, *, registry: TaskAdapterRegistry | None = None,
                 worker_factory: Callable = DockerWorker,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, poll_interval: float = 0.2):
        self._initialize(configuration, plan, registry, clock, sleep, poll_interval)
        self.policy, self.worker_factory = policy, worker_factory
        if not callable(worker_factory):
            raise ValueError("a Worker factory is required")
        baseline = InputTree.freeze(configuration.source_directory, "source")
        if baseline not in policy.sources.values() or policy.datasets.get(self.dataset.dataset_id) != self.public:
            raise ValueError("execution policy must register the original source and exact public dataset")

    def _initialize(self, configuration, plan, registry, clock, sleep, poll_interval):
        """Common frozen-input validation for the two explicit execution profiles."""
        self.config, self.plan = configuration, plan
        self.registry = registry if registry is not None else TaskAdapterRegistry()
        self.clock, self.sleep = clock, sleep
        output = configuration.output_directory.resolve()
        for protected in (configuration.source_directory.resolve(), configuration.dataset_directory.resolve()):
            if output == protected or output in protected.parents or protected in output.parents:
                raise ValueError("experiment output must not overlap registered source or data")
        if not 0 < poll_interval <= 10:
            raise ValueError("a bounded polling interval is required")
        self.poll_interval = poll_interval
        plan.validate_for_task(configuration.task)
        if plan.budget.max_job_seconds < 2 or plan.budget.max_total_job_seconds < 2:
            raise ValueError("the experiment budget must allow both training and prediction phases")
        if (set(plan.editable_files) != set(configuration.editable_files)
                or not set(plan.candidate_ids) <= set(configuration.candidate_ids)
                or not set(plan.seeds) <= set(configuration.seeds)
                or plan.primary_metric != configuration.primary_metric
                or set(plan.guardrail_metrics) != set(configuration.guardrail_metrics)):
            raise ValueError("research plan does not match configured task scope")
        if configuration.expected_data_hash is None:
            raise ValueError("pin the public data identity before executing an experiment")
        if not self.registry.can_score(configuration.evaluator_id):
            raise ValueError("the configured evaluator has no trusted scoring callback")
        self.public = InputTree.freeze(configuration.dataset_directory / "public", "public_data")
        self.dataset = self.registry.load_dataset(configuration.dataset_adapter, configuration.dataset_directory)
        self.public.verify()
        if (self.dataset.dataset_id != configuration.task.dataset_id
                or self.dataset.data_hash != configuration.expected_data_hash
                or Path(self.dataset.public).absolute() != self.public.directory.absolute()):
            raise ValueError("configured public dataset identity changed")
        description = self.registry.describe_evaluator(configuration.evaluator_id, self.dataset)
        self.public.verify()
        if description.protocol_hash != plan.protocol_hash:
            raise ValueError("plan and fixed evaluator protocols differ")
        if not {plan.primary_metric, *plan.guardrail_metrics} <= set(description.metrics):
            raise ValueError("plan requests unsupported evaluation metrics")
        baseline = InputTree.freeze(configuration.source_directory, "source")
        self.baseline_files = {
            path.relative_to(configuration.source_directory).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted(configuration.source_directory.rglob("*")) if path.is_file()
        }
        baseline.verify()
        if not set(configuration.editable_files) <= self.baseline_files.keys():
            raise ValueError("editable source files are not registered in the baseline")

    def _snapshot(self, snapshot: dict[str, Any]) -> tuple[InputTree, str]:
        source = InputTree.freeze(Path(snapshot["directory"]), "source")
        files = {name: confined_path(source.directory, name).read_text(encoding="utf-8")
                 for name in self.baseline_files}
        actual = {path.relative_to(source.directory).as_posix()
                  for path in source.directory.rglob("*") if path.is_file()}
        if actual != set(files) or set(snapshot["files"]) != actual:
            raise ValueError("experiment snapshot must retain the complete registered source tree")
        if content_hash(files) != snapshot["source_hash"]:
            raise ValueError("snapshot source_hash does not match its actual contents")
        for name, text in files.items():
            if name not in self.config.editable_files and text != self.baseline_files[name]:
                raise ValueError(f"read-only source dependency changed: {name}")
            if name.endswith(".py"):
                ast.parse(text, filename=name)
            elif name.endswith(".json"):
                read_json_strict(confined_path(source.directory, name), max_bytes=1_000_000)
        configuration = read_json_strict(
            confined_path(source.directory, f"configs/{self.config.config_name}.json"), max_bytes=1_000_000)
        source.verify()
        return source, content_hash(configuration)

    def _write(self, relative: str, value: Any) -> Path:
        path = confined_path(self.config.output_directory, relative)
        atomic_write(path, canonical_json(value).encode("utf-8"))
        return path

    def __call__(self, snapshot: dict[str, Any], *, candidate_id: str, seed: int,
                 max_seconds: int) -> ExperimentRecord:
        self.plan.check_scope(candidate_id, tuple(self.config.editable_files), seed)
        if type(max_seconds) is not int or not 2 <= max_seconds <= self.plan.budget.max_job_seconds:
            raise ValueError("experiment seconds must allow both phases and stay within the reviewed budget")
        source, config_hash = self._snapshot(snapshot)
        self.public.verify()
        experiment_id = "exp-" + content_hash({
            "plan": self.plan.plan_hash, "source": snapshot["source_hash"],
            "source_tree": source.sha256, "public_tree": self.public.sha256,
            "candidate": candidate_id, "seed": seed, "seconds": max_seconds,
            "policy": self.policy.identity(),
        })[:32]
        record = ExperimentRecord(
            experiment_id=experiment_id, plan_hash=self.plan.plan_hash,
            candidate_id=candidate_id, source_hash=snapshot["source_hash"], config_hash=config_hash,
            dataset_id=self.dataset.dataset_id, protocol_hash=self.plan.protocol_hash,
            image=self.policy.image, seed=seed, status="failed", program_validation="passed",
            failure="experiment has not executed",
        )
        with job_lock(self.config.output_directory, "experiment-controller") as acquired:
            if not acquired:
                raise RuntimeError("another experiment controller is using this task output directory")
            fence = confined_path(self.config.output_directory, "execution-blocked.json")
            if fence.exists():
                raise RuntimeError("an earlier job has unconfirmed stopping; operator/watchdog reconciliation is required")
            return self._execute(source, record, max_seconds)

    def _execute(self, source: InputTree, record: ExperimentRecord,
                 max_seconds: int) -> ExperimentRecord:
        started = self.clock()
        deadline = started + max_seconds
        relative = f"experiments/{record.experiment_id}"
        source_id = "source-" + source.sha256[:32]
        policy = replace(self.policy, sources={source_id: source},
                         datasets={self.dataset.dataset_id: self.public})
        state_root = confined_path(self.config.output_directory, f"{relative}/worker")
        policy_path = self._write(f"{relative}/worker-policy.json", {
            "image": policy.image, "gpu": policy.gpu, "max_limits": asdict(policy.max_limits),
            "max_output_bytes": policy.max_output_bytes, "max_concurrent_jobs": policy.max_concurrent_jobs,
            "sources": {key: {"directory": str(tree.directory), "sha256": tree.sha256}
                        for key, tree in policy.sources.items()},
            "datasets": {key: {"directory": str(tree.directory), "sha256": tree.sha256}
                         for key, tree in policy.datasets.items()},
        })
        # Stable limits preserve Worker job identities across a retry. Do not
        # derive predict.seconds from a fresh, variable remaining wall-clock time.
        train_seconds = min(policy.max_limits.seconds, max(1, max_seconds * 4 // 5))
        predict_seconds = min(policy.max_limits.seconds, max_seconds - max(1, max_seconds * 4 // 5))
        base = asdict(policy.max_limits)
        train_spec = JobSpec(source_id, self.dataset.dataset_id, config=self.config.config_name,
                             seed=record.seed, limits=ResourceLimits(**{**base, "seconds": train_seconds}))
        observations: list[dict[str, Any]] = []
        worker = None
        active_job = None
        uncertain_submission = False
        stage = "capabilities"

        def remaining():
            if self.clock() >= deadline:
                raise _PhaseFailure("experiment controller deadline exceeded", "timed_out")

        def wait_for(spec):
            nonlocal active_job, uncertain_submission, record
            remaining()
            observations.append({"operation": "submit", "spec": asdict(spec)})
            self._write(f"{relative}/execution.json", {"observations": observations})
            uncertain_submission = True
            submitted = worker.submit(spec)
            active_job = identifier(submitted["job_id"])
            uncertain_submission = False
            record = (replace(record, training_job_id=active_job) if spec.phase == "train" else
                      replace(record, prediction_job_id=active_job))
            while True:
                remaining()
                job = worker.poll(active_job)
                if job.get("job_id") != active_job:
                    raise RuntimeError("Worker returned a mismatched job identity")
                status = job.get("status")
                if status in TERMINAL:
                    observations.append({"operation": "terminal", "job_id": active_job,
                                         "status": status, "exit_code": job.get("exit_code")})
                    completed_job = active_job
                    if status != "completed" or type(job.get("exit_code")) is not int or job["exit_code"] != 0:
                        # A missing/stopped/cancelled observation is not by
                        # itself proof that a late or restarted job cannot run.
                        if status == "failed" and type(job.get("exit_code")) is int and job["exit_code"] != 0:
                            active_job = None
                        raise _PhaseFailure(f"{spec.phase} job {completed_job} ended as {status}",
                                            status if status != "completed" else "failed")
                    active_job = None
                    return completed_job
                self.sleep(min(self.poll_interval, max(0, deadline - self.clock())))

        try:
            worker = self.worker_factory(state_root, policy)
            worker.capabilities()
            remaining()
            stage = "train"
            training_id = wait_for(train_spec)
            record = replace(record, training_job_id=training_id)
            stage = "predict"
            prediction_spec = JobSpec(source_id, self.dataset.dataset_id, phase="predict",
                                      config=self.config.config_name, seed=record.seed,
                                      limits=ResourceLimits(**{**base, "seconds": predict_seconds}),
                                      model_job_id=training_id)
            prediction_id = wait_for(prediction_spec)
            record = replace(record, prediction_job_id=prediction_id)
            stage = "collect"
            predictions = worker.collect_predictions(prediction_id)
            collected_hash = content_hash(predictions)
            remaining()
            source.verify()
            self.public.verify()
            stage = "score"
            grade = self.registry.score_predictions(self.config.evaluator_id, self.dataset, predictions)
            remaining()
            if content_hash(predictions) != collected_hash:
                raise ValueError("scorer modified the collected predictions")
            self.public.verify()
            source.verify()
            if grade["protocol_hash"] != self.plan.protocol_hash:
                raise ValueError("independent score does not match the reviewed protocol")
            prediction_path = self._write(f"{relative}/predictions.json", predictions)
            grade_path = self._write(f"{relative}/grade.json", grade)
            score = EvidenceRef(
                source=f"{relative}/grade.json", sha256=hashlib.sha256(grade_path.read_bytes()).hexdigest(),
                location=f"/metrics/{self.plan.primary_metric}",
                conditions="Independent development scoring from collected Worker predictions; not final-test evidence.")
            record = replace(record, status="completed", failure=None, score=score)
            observations.append({"operation": "score", "predictions_sha256":
                                 hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
                                 "grade_sha256": score.sha256})
        except (Exception, KeyboardInterrupt) as error:
            status = error.status if isinstance(error, _PhaseFailure) else "failed"
            record = replace(record, status=status, failure=f"{stage}: {type(error).__name__}: {error}")
            if active_job is not None or uncertain_submission:
                cancellation: dict[str, Any] = {"job_id": active_job, "stop_confirmed": False,
                                                "continued_watchdog_required": True}
                if active_job is not None and worker is not None:
                    try:
                        cancellation["worker_observation"] = worker.cancel(active_job)
                    except Exception as stop_error:
                        cancellation["error"] = f"{type(stop_error).__name__}: {stop_error}"
                # A cancel response or missing container is not a fresh stop proof.
                # Block later experiments until the operator reconciles the job.
                observations.append({"operation": "cancel", **cancellation})
                self._write("execution-blocked.json", {"experiment_id": record.experiment_id,
                                                        "worker_state": str(state_root),
                                                        "worker_policy": str(policy_path), **cancellation})
            if isinstance(error, KeyboardInterrupt):
                self._write(f"{relative}/record.json", record.to_dict())
                raise
        finally:
            self._write(f"{relative}/execution.json", {
                "observations": observations, "elapsed_seconds": self.clock() - started,
                "worker_state": str(state_root), "worker_policy": str(policy_path),
                "supervisor_required": True,
                "hard_overall_timeout_verified": False,
            })
        self._write(f"{relative}/record.json", record.to_dict())
        return record
