"""The common experiment pipeline composed from the small SSH/tmux tools."""
from __future__ import annotations
from dataclasses import replace
import hashlib
from pathlib import PurePosixPath
import time
from reclab_agent.execution.locking import job_lock
from reclab_agent.domain.contracts import content_hash
from reclab_agent.infrastructure.files import confined_path
from reclab_agent.tools.remote import RemoteTools
from reclab_agent.tools.contracts import ToolError
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.research import EvidenceRef, ExperimentRecord
from reclab_agent.experiments.runner import ExperimentRunner, TERMINAL, _PhaseFailure


class RemoteExperimentRunner(ExperimentRunner):
    """Train, predict, download and locally score; no model-specific callback.

    This explicit native profile is for reviewed code on the user's dedicated
    GPU. tmux and path checks are process management, not container isolation.
    The server helper enforces each phase's timeout even if this client exits.
    """

    def __init__(self, configuration, plan, tools: RemoteTools, *, registry=None,
                 clock=time.monotonic, sleep=time.sleep, poll_interval=2):
        self._initialize(configuration, plan, registry, clock, sleep, poll_interval)
        if configuration.execution_backend != "ssh_tmux":
            raise ValueError("native execution requires the explicit ssh_tmux task profile")
        self.tools = tools

    def __call__(self, snapshot, *, candidate_id, seed, max_seconds):
        self.plan.check_scope(candidate_id, tuple(self.config.editable_files), seed)
        if type(max_seconds) is not int or not 2 <= max_seconds <= self.plan.budget.max_job_seconds:
            raise ValueError("experiment time must allow both phases within the reviewed budget")
        source, config_hash = self._snapshot(snapshot)
        self.public.verify()
        with job_lock(self.config.output_directory, "experiment-controller") as acquired:
            if not acquired:
                raise RuntimeError("another experiment controller is using the task output")
            if confined_path(self.config.output_directory, "execution-blocked.json").exists():
                raise RuntimeError("an earlier job has unconfirmed stopping; reconcile it before retrying")
            environment = self.tools.bootstrap()
            marker = confined_path(self.config.output_directory, "active-remote-job.json")
            prior = read_json_strict(marker, max_bytes=1_000_000) if marker.exists() else None
            connection = self.tools.connection
            if prior and (prior.get("remote_root"), prior.get("host"), prior.get("port")) != (
                    connection.remote_root, connection.host, connection.port):
                raise RuntimeError("an earlier remote job belongs to a different connection; reconcile it first")
            if marker.exists() and not self.config.resume_jobs:
                # A controller restart conservatively stops its previous phase
                # before admitting another experiment. Raw job_status remains
                # available for inspecting a running tmux job without doing this.
                try:
                    stopped = self.tools.stop_job(prior["job_id"])
                    if stopped.get("stop_confirmed") is not True:
                        raise RuntimeError("remote stop has not been confirmed")
                except Exception as error:
                    self._write("execution-blocked.json", {**prior, "stop_confirmed": False,
                                                           "error": str(error)})
                    raise RuntimeError("prior remote job cannot be reconciled; new experiments are blocked") from error
                self._write(f"experiments/{prior['experiment_id']}/recovery-stop.json", stopped)
                marker.unlink()
            connection = self.tools.connection
            details = {"environment": environment, "host": connection.host, "port": connection.port,
                       "remote_root": connection.remote_root}
            runtime = {"backend": "ssh_tmux", "details": details, "fingerprint": content_hash(details)}
            experiment_id = "exp-" + content_hash({
                "plan": self.plan.plan_hash, "source": snapshot["source_hash"],
                "source_tree": source.sha256, "public_tree": self.public.sha256,
                "candidate": candidate_id, "seed": seed, "seconds": max_seconds,
                "runtime": runtime["fingerprint"],
            })[:32]
            relative = f"experiments/{experiment_id}"
            recovering = bool(prior and self.config.resume_jobs)
            if recovering and (prior.get("experiment_id") != experiment_id or prior.get("job_id") not in {
                    experiment_id + "-train", experiment_id + "-predict"}):
                raise RuntimeError("pending remote job does not match this experiment; reconcile it first")
            existing = confined_path(self.config.output_directory, f"{relative}/record.json")
            if existing.exists():
                saved = read_json_strict(existing, max_bytes=1_000_000)
                if not recovering or saved["status"] == "completed":
                    return ExperimentRecord.from_dict(saved)
                self._write(f"{relative}/recovery-previous-record.json", saved)
            record = ExperimentRecord(
                experiment_id, self.plan.plan_hash, candidate_id, snapshot["source_hash"],
                config_hash, self.dataset.dataset_id, self.plan.protocol_hash, None, seed,
                status="failed", program_validation="passed", failure="experiment has not executed",
                runtime=runtime,
            )
            return self._remote_execute(source, record, relative, max_seconds)

    def _remote_execute(self, source, record, relative, max_seconds):
        started = self.clock()
        deadline = started + max_seconds
        observations = []
        active_job = None
        stage = "upload"
        source_remote = "inputs/source-" + source.sha256
        public_remote = "inputs/public-" + self.public.sha256
        root = self.tools.connection.remote_root.rstrip("/")
        remote_output = f"results/{record.experiment_id}"
        marker = confined_path(self.config.output_directory, "active-remote-job.json")
        prior = read_json_strict(marker, max_bytes=1_000_000) if self.config.resume_jobs and marker.exists() else None
        if prior:
            active_job = prior["job_id"]

        def remaining():
            if self.clock() >= deadline:
                raise _PhaseFailure("experiment controller deadline exceeded", "timed_out")

        def persist():
            self._write(f"{relative}/execution.json", {
                "backend": "ssh_tmux", "observations": observations,
                "elapsed_seconds": self.clock() - started,
                "remote_root": root, "remote_output": remote_output,
                "per_phase_timeout_enforced_remotely": True,
                "container_isolation": False,
            })

        def wait_for(phase, seconds, extra):
            nonlocal active_job, record
            remaining()
            active_job = record.experiment_id + "-" + phase
            record = replace(record, **{("training_job_id" if phase == "train" else
                                         "prediction_job_id"): active_job})
            # Persist the known ID before submitting: a lost SSH reply is not a
            # safe reason to submit a new job or assume nothing started.
            observations.append({"operation": "submit", "phase": phase, "job_id": active_job})
            self._write("active-remote-job.json", {"experiment_id": record.experiment_id,
                        "job_id": active_job, "remote_root": self.tools.connection.remote_root,
                        "host": self.tools.connection.host, "port": self.tools.connection.port})
            persist()
            argv = [self.tools.connection.python_executable, "-B", "-u",
                    f"{root}/{source_remote}/src/{phase}.py",
                    "--data", f"{root}/{public_remote}",
                    "--output", f"{root}/{remote_output}/{phase}",
                    "--config", f"{root}/{source_remote}/configs/{self.config.config_name}.json", *extra]
            # Stable phase limits keep retries of this job ID identical. The
            # local deadline also counts transfer/control time and stops a job
            # when necessary; the remote limit survives loss of the client.
            if self.config.resume_jobs:
                try:
                    job = self.tools.job_status(active_job)
                except KeyError:
                    job = self.tools.run_remote(active_job, argv, source_remote, seconds, record.seed)
                except ToolError as error:
                    missing = error.code == "NOT_FOUND" or (
                        error.code == "INVALID_ARGUMENTS" and str(error) == "unknown job_id")
                    if not missing:
                        raise
                    job = self.tools.run_remote(active_job, argv, source_remote, seconds, record.seed)
            else:
                job = self.tools.run_remote(active_job, argv, source_remote, seconds, record.seed)
            while True:
                if job.get("job_id") != active_job:
                    raise ValueError("remote job identity mismatch")
                if job.get("status") in TERMINAL:
                    observations.append({"operation": "terminal", "phase": phase, **job})
                    self._write(f"{relative}/{phase}-log.json", self.tools.tail_log(active_job, 200))
                    if job["status"] != "completed" or type(job.get("exit_code")) is not int or job["exit_code"] != 0:
                        raise _PhaseFailure(f"{phase} ended as {job['status']}", job["status"]
                                            if job["status"] != "completed" else "failed")
                    if job.get("stop_confirmed") is not True:
                        raise RuntimeError("completed phase has not confirmed process-group cleanup")
                    marker.unlink(missing_ok=True)
                    active_job = None
                    return
                remaining()
                self.sleep(min(self.poll_interval, max(0, deadline - self.clock())))
                job = self.tools.job_status(active_job)

        try:
            # Only the frozen source and public files enter the upload roots.
            source_local = source.directory.relative_to(self.config.output_directory / "snapshots").as_posix()
            for local_tree, label, prefix, base in ((source, "source", source_remote, source_local),
                                                    (self.public, "public", public_remote, "")):
                if prior:
                    # A persisted phase is submitted only after every upload.
                    # Recheck hashes during collection; do not rewrite live inputs.
                    observations.append({"operation": "reuse_uploaded_inputs", "root": prefix})
                    continue
                for path in sorted(local_tree.directory.rglob("*")):
                    if path.is_file():
                        remaining()
                        name = path.relative_to(local_tree.directory).as_posix()
                        local_name = str(PurePosixPath(base) / name) if base else name
                        info = self.tools.upload(label, local_name, prefix + "/" + name)
                        observations.append({"operation": "upload", "path": prefix + "/" + name, **info})
            source.verify()
            self.public.verify()
            stage = "train"
            train_seconds = max(1, max_seconds * 4 // 5)
            if prior and prior["job_id"].endswith("-predict"):
                training_id = record.experiment_id + "-train"
                training = self.tools.job_status(training_id)
                if training.get("status") != "completed" or training.get("exit_code") != 0 or training.get("stop_confirmed") is not True:
                    raise RuntimeError("prediction recovery requires a verified completed training job")
                record = replace(record, training_job_id=training_id)
            else:
                wait_for("train", train_seconds, [])
            stage = "predict"
            wait_for("predict", max_seconds - train_seconds,
                     ["--model", f"{root}/{remote_output}/train", "--split", "valid"])
            stage = "collect"
            # Verify the remote inputs after candidate execution as well.
            for tree, prefix in ((source, source_remote), (self.public, public_remote)):
                for path in sorted(tree.directory.rglob("*")):
                    if path.is_file():
                        name = path.relative_to(tree.directory).as_posix()
                        observed = self.tools.file_stat(prefix + "/" + name)
                        if observed["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                            raise ValueError("remote source/public inputs changed during execution")
            for name in ("predict/predictions.json", "predict/prediction.json", "train/training.json", "train/epochs.jsonl",
                         "train/best.pt", "train/smoke.json", "predict/process_metrics.json"):
                remaining()
                try:
                    self.tools.download(f"{remote_output}/{name}", f"{relative}/{name}")
                except ToolError as error:
                    if name == "predict/predictions.json" or error.code != "NOT_FOUND":
                        raise
                    observations.append({"operation": "optional_artifact_absent", "path": name})
            predictions = read_json_strict(confined_path(self.config.output_directory,
                                                        f"{relative}/predict/predictions.json"))
            before = content_hash(predictions)
            stage = "score"
            remaining()
            grade = self.registry.score_predictions(self.config.evaluator_id, self.dataset, predictions)
            source.verify()
            self.public.verify()
            if content_hash(predictions) != before or grade["protocol_hash"] != self.plan.protocol_hash:
                raise ValueError("independent scoring inputs or protocol changed")
            grade_path = self._write(f"{relative}/grade.json", grade)
            score = EvidenceRef(f"{relative}/grade.json", hashlib.sha256(grade_path.read_bytes()).hexdigest(),
                                f"/metrics/{self.plan.primary_metric}",
                                "Independent local validation-set evaluation of remote predictions under the recorded training seed.")
            record = replace(record, status="completed", score=score, failure=None)
            observations.append({"operation": "score", "grade_sha256": score.sha256})
        except (Exception, KeyboardInterrupt) as error:
            status = error.status if isinstance(error, _PhaseFailure) else "failed"
            record = replace(record, status=status, failure=f"{stage}: {type(error).__name__}: {error}")
            if active_job is not None:
                cancellation = {"job_id": active_job, "stop_confirmed": False}
                try:
                    stopped = self.tools.stop_job(active_job)
                    cancellation.update(stopped)
                except Exception as stop_error:
                    cancellation["error"] = f"{type(stop_error).__name__}: {stop_error}"
                observations.append({"operation": "stop", **cancellation})
                if cancellation.get("stop_confirmed") is not True:
                    self._write("execution-blocked.json", {"experiment_id": record.experiment_id,
                                                           "remote_root": root, **cancellation})
                else:
                    marker.unlink(missing_ok=True)
            self._write(f"{relative}/record.json", record.to_dict())
            if isinstance(error, KeyboardInterrupt):
                raise
        finally:
            persist()
        self._write(f"{relative}/record.json", record.to_dict())
        return record
