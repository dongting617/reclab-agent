"""Fail-closed, daemon-local Docker jobs. Candidate code is never imported here.

The registry/policy is administrator configuration, not an Agent tool argument.
Only a registered source ID, public dataset ID and bounded job parameters cross
the tool boundary. Docker access is a privileged capability of this worker.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import uuid
from reclab_agent.domain.contracts import canonical_json, content_hash, identifier
from reclab_agent.infrastructure.files import (
    atomic_write,
    confined_path,
    read_regular_file as _regular,
    validated_directory as _directory,
)
from reclab_agent.execution.docker import SubprocessDocker
from reclab_agent.execution.locking import job_lock
from reclab_agent.execution.models import (
    CapabilityError,
    CommandResult,
    DockerError,
    InputTree,
    JobSpec,
    OwnershipError,
    ResourceLimits,
    WorkerPolicy,
)


def _serialized_job(*, busy: str = "state"):
    """Fence Docker side effects with an OS lock, not an expiring DB lease alone."""
    def decorate(method):
        @wraps(method)
        def serialized(self, job_id, *args, **kwargs):
            with self._job_lock(job_id) as acquired:
                if not acquired:
                    if busy == "raise":
                        raise DockerError("job is being reconciled by another live worker; retry collection")
                    return self.get(job_id)
                return method(self, job_id, *args, **kwargs)
        return serialized
    return decorate


class DockerWorker:
    TERMINAL = {"completed", "failed", "timed_out", "cancelled", "lost"}
    LABEL_PREFIX = "io.reclab."

    def __init__(self, root: str | Path, policy: WorkerPolicy, *,
                 runner: Callable[..., CommandResult] | None = None,
                 clock: Callable[[], float] = time.time,
                 initialize: bool = True, database_timeout: float = 30.0):
        self.root = _directory(root, create=initialize)
        if initialize and sys.platform == "linux":
            os.chmod(self.root, 0o700)
        self.policy, self.clock = policy, clock
        self.database = confined_path(self.root, "jobs.sqlite3")
        self.runner = runner or SubprocessDocker(self.root)
        self.session = uuid.uuid4().hex
        self.database_timeout = database_timeout
        if not initialize:
            # Stop-only supervisors must not request a schema write lock during
            # startup. Schema migration remains an explicit Worker operation.
            with self._connect() as db:
                row = db.execute("SELECT value FROM worker_meta WHERE key='owner'").fetchone()
                if row is None:
                    raise ValueError("existing Worker owner identity is required")
                db.execute("SELECT job_id FROM docker_stop_intents LIMIT 0")
                self.owner = row[0]
            return
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS worker_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS docker_jobs (
                    job_id TEXT PRIMARY KEY, job_key TEXT UNIQUE NOT NULL, spec TEXT NOT NULL,
                    policy_hash TEXT NOT NULL, status TEXT NOT NULL, container TEXT NOT NULL UNIQUE,
                    created REAL NOT NULL, deadline REAL, lease_owner TEXT, lease_until REAL,
                    stop_reason TEXT, exit_code INTEGER, reason TEXT, receipt TEXT);
                CREATE TABLE IF NOT EXISTS docker_stop_intents (
                    job_id TEXT PRIMARY KEY REFERENCES docker_jobs(job_id),
                    reason TEXT NOT NULL CHECK(reason IN ('timed_out','cancelled')),
                    requested REAL NOT NULL, last_checked REAL,
                    container_state TEXT, confirmed REAL, detail TEXT);
            """)
            db.execute("INSERT OR IGNORE INTO worker_meta VALUES ('owner', ?)", (uuid.uuid4().hex,))
            self.owner = db.execute("SELECT value FROM worker_meta WHERE key='owner'").fetchone()[0]

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=self.database_timeout)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def _call(self, args: list[str]) -> CommandResult:
        return self.runner(args, timeout=10)

    @contextmanager
    def _job_lock(self, job_id: str):
        """Nonblocking, process-safe and same-thread reentrant job lock.

        The lock file is never unlinked. A suspended live launcher retains its
        lock even after its database lease expires; a crashed process releases
        the OS lock automatically. This fences create/start/kill side effects.
        """
        with job_lock(self.root, job_id) as acquired:
            yield acquired

    def capabilities(self) -> dict[str, Any]:
        try:
            result = self._call(["info", "--format", "{{json .}}"])
        except (DockerError, CapabilityError) as error:
            raise CapabilityError(str(error)) from error
        if result.returncode:
            raise CapabilityError("local Docker daemon is unavailable; no host execution fallback")
        try:
            info = json.loads(result.stdout)
        except (ValueError, TypeError) as error:
            raise CapabilityError("Docker returned invalid capability information") from error
        if not isinstance(info, dict) or info.get("OSType") != "linux":
            raise CapabilityError("a Linux Docker daemon is required")
        seccomp = [str(option).split(",") for option in info.get("SecurityOptions", [])
                   if "name=seccomp" in str(option).split(",")]
        if not seccomp or any("profile=unconfined" in parts for parts in seccomp):
            raise CapabilityError("Docker daemon must report an enabled seccomp profile")
        if not all(info.get(name) is True for name in ("MemoryLimit", "SwapLimit", "CpuCfsPeriod", "CpuCfsQuota", "PidsLimit")):
            raise CapabilityError("Docker daemon cannot enforce every required CPU/memory/swap/PID limit")
        if self.policy.gpu and "nvidia" not in info.get("Runtimes", {}):
            raise CapabilityError("GPU policy requires a configured NVIDIA container runtime")
        image = self._call(["image", "inspect", "--", self.policy.image])
        if image.returncode:
            raise CapabilityError("approved image digest is not present locally; worker never pulls images")
        try:
            images = json.loads(image.stdout)
            present = (len(images) == 1 and self.policy.image in images[0].get("RepoDigests", [])
                       and not images[0].get("Config", {}).get("Volumes"))
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            present = False
        if not present:
            raise CapabilityError("local image does not match the approved immutable digest")
        return {"docker": True, "linux": True, "image": self.policy.image,
                "gpu_requested": self.policy.gpu, "real_container_tested": False}

    def _validate(self, spec: JobSpec) -> tuple[InputTree, InputTree]:
        if not isinstance(spec, JobSpec):
            raise ValueError("JobSpec is required")
        try:
            source, data = self.policy.sources[spec.source_id], self.policy.datasets[spec.dataset_id]
        except KeyError as error:
            raise ValueError("unregistered source or public dataset ID") from error
        for name in ("cpus", "memory_mb", "pids", "seconds"):
            if getattr(spec.limits, name) > getattr(self.policy.max_limits, name):
                raise ValueError("job exceeds administrator resource policy")
        source.verify()
        data.verify()
        if not confined_path(source.directory, f"configs/{spec.config}.json").is_file():
            raise ValueError("registered config does not exist")
        if spec.phase == "predict":
            model = self.get(spec.model_job_id)
            training = JobSpec.from_dict(model["spec"])
            if (model["status"] != "completed" or training.phase != "train"
                    or training.source_id != spec.source_id or training.dataset_id != spec.dataset_id
                    or training.config != spec.config or training.seed != spec.seed
                    or model["policy_hash"] != self.policy.identity() or model["job_key"] != self._key(training)):
                raise ValueError("prediction requires a completed, identical registered training job")
            self._verify_receipt(model)
        return source, data

    def _key(self, spec: JobSpec) -> str:
        return content_hash({"spec": asdict(spec), "source": self.policy.sources[spec.source_id].sha256,
                            "data": self.policy.datasets[spec.dataset_id].sha256, "policy": self.policy.identity()})

    def submit(self, spec: JobSpec) -> dict[str, Any]:
        self._validate(spec)
        key = self._key(spec)
        job_id = f"job-{key[:40]}"
        with self._connect() as db:
            existing = db.execute("SELECT job_id FROM docker_jobs WHERE job_key=?", (key,)).fetchone()
        if existing:
            return self.poll(existing[0])
        self.capabilities()  # Check before creating a job; no host subprocess fallback.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO docker_jobs(job_id,job_key,spec,policy_hash,status,container,created) "
                       "VALUES(?,?,?,?,'queued',?,?)", (job_id, key, canonical_json(asdict(spec)), self.policy.identity(),
                                                      f"reclab-{key[:40]}", self.clock()))
        return self._launch(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        identifier(job_id)
        with self._connect() as db:
            row = db.execute("SELECT * FROM docker_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        value = dict(row)
        value["spec"] = json.loads(value["spec"])
        value["receipt"] = json.loads(value["receipt"]) if value["receipt"] else None
        return value

    def _update(self, job_id: str, **fields: Any):
        allowed = {"status", "deadline", "lease_owner", "lease_until", "stop_reason", "exit_code", "reason", "receipt"}
        if not fields.keys() <= allowed:
            raise ValueError("invalid job update")
        with self._connect() as db:
            # Terminal rows are immutable even if a poll/cancel races launch completion.
            db.execute("UPDATE docker_jobs SET " + ",".join(f"{name}=?" for name in fields)
                       + " WHERE job_id=? AND status NOT IN ('completed','failed','timed_out','cancelled','lost')",
                       (*fields.values(), job_id))

    def _request_stop(self, job_id: str, reason: str) -> str:
        """Durable, first-writer-wins stop intent; deliberately independent of job locks.

        A stop-only watchdog must be able to write this while a suspended
        launcher owns the OS job lock. The tombstone is retained after terminal
        state so a late Docker start can still be detected and stopped.
        """
        identifier(job_id)
        if reason not in {"timed_out", "cancelled"}:
            raise ValueError("unsupported stop reason")
        # Separate, private journal is not subject to SQLite's write lock. It
        # fences a resumed launcher even if SQLite accounting is temporarily busy.
        reason = self._journal_stop(job_id, reason)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT stop_reason FROM docker_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            db.execute("INSERT OR IGNORE INTO docker_stop_intents(job_id,reason,requested) VALUES(?,?,?)",
                       (job_id, row["stop_reason"] or reason, self.clock()))
            chosen = db.execute("SELECT reason FROM docker_stop_intents WHERE job_id=?", (job_id,)).fetchone()[0]
            db.execute("UPDATE docker_jobs SET status='stopping',stop_reason=? WHERE job_id=? "
                       "AND status NOT IN ('completed','failed','timed_out','cancelled','lost')", (chosen, job_id))
        return chosen

    def _journal_reason(self, job_id: str) -> str | None:
        identifier(job_id)
        markers = []
        for reason in ("cancelled", "timed_out"):
            path = confined_path(self.root, f"stop-journal/{job_id}.{reason}")
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size:
                raise ValueError("stop journal marker must be an empty, single-link regular file")
            markers.append((info.st_mtime_ns, reason))
        return min(markers)[1] if markers else None

    def _journal_stop(self, job_id: str, reason: str) -> str:
        identifier(job_id)
        if reason not in {"timed_out", "cancelled"}:
            raise ValueError("unsupported stop reason")
        existing = self._journal_reason(job_id)
        if existing:
            return existing
        directory = _directory(confined_path(self.root, "stop-journal"), create=True)
        path = confined_path(directory, f"{job_id}.{reason}")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError:
            pass
        else:
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if sys.platform == "linux":
                descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        return self._journal_reason(job_id)

    def _stop_reason(self, job_id: str) -> str | None:
        journal = self._journal_reason(job_id)
        if journal:
            return journal
        with self._connect() as db:
            row = db.execute("SELECT reason FROM docker_stop_intents WHERE job_id=?", (job_id,)).fetchone()
        return row[0] if row else self.get(job_id)["stop_reason"]

    def _labels(self, job: dict[str, Any]) -> dict[str, str]:
        return {"io.reclab.worker": self.owner, "io.reclab.job": job["job_id"], "io.reclab.spec": job["job_key"]}

    def _inspect(self, job: dict[str, Any], *, reference: str | None = None) -> dict[str, Any] | None:
        result = self._call(["container", "inspect", "--", reference or job["container"]])
        if result.returncode:
            if any(message in result.stderr.lower() for message in ("no such object", "no such container")):
                return None
            raise DockerError("container inspect failed; execution outcome is uncertain")
        try:
            values = json.loads(result.stdout)
            if len(values) != 1 or not isinstance(values[0], dict):
                raise ValueError()
            container = values[0]
            labels = container["Config"].get("Labels", {}) or {}
            if not isinstance(labels, dict):
                raise ValueError("invalid ownership labels")
            if (not isinstance(container.get("State"), dict)
                    or container["State"].get("Status") not in {"created", "running", "paused", "restarting", "removing", "exited", "dead"}):
                raise ValueError("invalid container State")
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
            raise DockerError("invalid container inspection") from error
        if any(labels.get(key) != value for key, value in self._labels(job).items()):
            raise OwnershipError("container name belongs to an unknown job; it will not be started, stopped or removed")
        return container

    def _container_id(self, container: dict[str, Any]) -> str:
        reference = container.get("Id")
        if not isinstance(reference, str) or not re.fullmatch(r"[a-f0-9]{64}", reference):
            raise DockerError("owned container inspection is missing its immutable ID")
        return reference

    def _output(self, job_id: str) -> Path:
        identifier(job_id)
        output = confined_path(self.root, f"outputs/{job_id}")
        return _directory(output, create=True)

    def _create_args(self, job: dict[str, Any]) -> list[str]:
        spec = JobSpec.from_dict(job["spec"])
        source, data = self._validate(spec)
        output = self._output(job["job_id"])
        if sys.platform == "linux":
            # A non-root operator in the Docker group need not chown. The private
            # state-root (0700) protects this host path; only this output is mounted.
            os.chmod(output, 0o777)
        args = ["container", "create", "--name", job["container"], "--pull", "never",
                "--user", "65532:65532", "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges=true", "--cpus", str(spec.limits.cpus),
                "--memory", f"{spec.limits.memory_mb}m", "--memory-swap", f"{spec.limits.memory_mb}m",
                "--pids-limit", str(spec.limits.pids), "--ulimit", "nofile=256:256",
                "--init", "--ipc", "private", "--workdir", "/workspace", "--log-driver", "local",
                "--log-opt", "max-size=1m", "--log-opt", "max-file=1",
                "--log-opt", "compress=false",
                "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777",
                "--mount", f"type=bind,source={source.directory},target=/workspace,readonly",
                "--mount", f"type=bind,source={data.directory},target=/data,readonly",
                "--mount", f"type=bind,source={output},target=/output",
                "--env", f"RECLAB_SEED={spec.seed}", "--env", "HOME=/tmp",
                "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1"]
        for key, value in self._labels(job).items():
            args += ["--label", f"{key}={value}"]
        if self.policy.gpu:
            args += ["--gpus", "device=0"]  # Operator capability, never a model-selected option.
        if spec.phase == "predict":
            self._verify_receipt(self.get(spec.model_job_id))
            args += ["--mount", f"type=bind,source={self._output(spec.model_job_id)},target=/model,readonly"]
        # Fixed Python entrypoint and fixed argv; source code stays inside the container.
        args += ["--entrypoint", "/usr/local/bin/python", self.policy.image, "-E", "-s", "-B",
                 f"/workspace/src/{'train' if spec.phase == 'train' else 'predict'}.py",
                 "--data", "/data", "--output", "/output", "--config", f"/workspace/configs/{spec.config}.json"]
        if spec.phase == "predict":
            args += ["--model", "/model", "--split", "valid"]
        return args

    @_serialized_job()
    def _launch(self, job_id: str) -> dict[str, Any]:
        now = self.clock()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM docker_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] in self.TERMINAL or row["status"] in {"running", "stopping"}:
                return self.get(job_id)
            if row["policy_hash"] != self.policy.identity():
                raise ValueError("cannot resume a job under a changed worker policy")
            if row["job_key"] != self._key(JobSpec.from_dict(json.loads(row["spec"]))):
                raise ValueError("registered job inputs changed; restore original frozen identities")
            if row["lease_until"] and row["lease_until"] > now:
                return self.get(job_id)
            others = db.execute("SELECT COUNT(*) FROM docker_jobs WHERE job_id<>? AND status IN "
                                "('launching','running','stopping','reconciling')", (job_id,)).fetchone()[0]
            if others >= self.policy.max_concurrent_jobs:
                return self.get(job_id)
            seconds = json.loads(row["spec"])["limits"]["seconds"]
            db.execute("UPDATE docker_jobs SET status='launching',lease_owner=?,lease_until=?,deadline=COALESCE(deadline,?) "
                       "WHERE job_id=?", (self.session, now + 60, now + seconds, job_id))
        job = self.get(job_id)
        try:
            existing = self._inspect(job)
            requested_stop = self._stop_reason(job_id)
            if requested_stop:
                return self._terminate(job_id, requested_stop, launching=True)
            if self.clock() >= job["deadline"]:
                if existing is None:
                    self._request_stop(job_id, "timed_out")
                    self._update(job_id, status="timed_out", reason="deadline elapsed before container creation")
                else:
                    return self._terminate(job_id, "timed_out", launching=True)
            else:
                if existing is None:
                    result = self._call(self._create_args(job))
                    # A timeout/nonzero may happen after remote creation. Inspect deterministic identity.
                    existing = self._inspect(job)
                    if existing is None:
                        raise DockerError("Docker create did not produce an identifiable container")
                    if result.returncode:
                        self._update(job_id, reason="create returned an error; matching owned container recovered")
                requested_stop = self._stop_reason(job_id)
                if requested_stop:
                    return self._terminate(job_id, requested_stop, launching=True)
                if self.clock() >= job["deadline"]:
                    return self._terminate(job_id, "timed_out", launching=True)
                state = existing.get("State", {}).get("Status")
                if state == "created":
                    self._call(["container", "start", "--", job["container"]])
                    existing = self._inspect(job)
                    if existing and existing.get("State", {}).get("Status") == "created":
                        raise DockerError("container start is not confirmed; preserving the existing created container")
                if existing is None:
                    raise DockerError("owned container disappeared during launch")
                requested_stop = self._stop_reason(job_id)
                if requested_stop:
                    return self._terminate(job_id, requested_stop, launching=True)
                self._update(job_id, status="running", reason=None)
                return self._observe(job_id, existing)
        except OwnershipError:
            raise
        except (DockerError, CapabilityError) as error:
            self._update(job_id, status="stopping" if self.get(job_id)["stop_reason"] else "reconciling", reason=str(error))
        finally:
            with self._connect() as db:
                db.execute("UPDATE docker_jobs SET lease_owner=NULL,lease_until=NULL WHERE job_id=? AND lease_owner=?",
                           (job_id, self.session))
        return self.get(job_id)

    @_serialized_job()
    def _observe(self, job_id: str, container: dict[str, Any]) -> dict[str, Any]:
        job = self.get(job_id)
        state = container.get("State", {})
        if state.get("Status") in {"exited", "dead"}:
            code = state.get("ExitCode")
            if type(code) is not int:
                raise DockerError("container exit code is missing")
            finished = self.clock()
            if isinstance(state.get("FinishedAt"), str):
                try:
                    finished = datetime.fromisoformat(state["FinishedAt"].replace("Z", "+00:00")).timestamp()
                except ValueError:
                    raise DockerError("container finish timestamp is invalid")
            overdue = job["deadline"] is not None and finished > job["deadline"]
            status = job["stop_reason"] or ("timed_out" if overdue else ("completed" if code == 0 and not state.get("OOMKilled") else "failed"))
            if status in {"timed_out", "cancelled"}:
                status = self._request_stop(job_id, status)
            reason = "oom_killed" if state.get("OOMKilled") else (None if code == 0 else "nonzero_exit")
            receipt = None
            if status == "completed":
                try:
                    manifest = self._manifest(job_id)
                    if not manifest["files"] or (job["spec"]["phase"] == "predict" and "predictions.json" not in manifest["files"]):
                        raise ValueError("required output artifacts are missing")
                    receipt = canonical_json(manifest)
                except (OSError, ValueError) as error:
                    status, reason = "failed", f"invalid_output: {error}"
            self._update(job_id, status=status, exit_code=code, reason=reason, receipt=receipt)
        elif job["deadline"] is not None and self.clock() >= job["deadline"]:
            return self._terminate(job_id, "timed_out")
        return self.get(job_id)

    @_serialized_job()
    def poll(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in self.TERMINAL:
            return job
        if job["policy_hash"] != self.policy.identity():
            raise ValueError("restore the original security policy to resume; explicit cancellation remains available")
        if job["status"] in {"queued", "launching", "reconciling"}:
            return self._launch(job_id)
        try:
            container = self._inspect(job)
            if container is None:
                self._update(job_id, status="lost", reason="previously launched container is missing; automatic relaunch forbidden")
                return self.get(job_id)
            if job["stop_reason"]:
                return self._terminate(job_id, job["stop_reason"])
            return self._observe(job_id, container)
        except OwnershipError:
            raise
        except (DockerError, CapabilityError) as error:
            self._update(job_id, reason=str(error))
            return self.get(job_id)

    @_serialized_job()
    def _terminate(self, job_id: str, reason: str, *, launching: bool = False) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in self.TERMINAL:
            return job
        reason = self._request_stop(job_id, reason)
        job = self.get(job_id)
        if not launching and job["lease_until"] and job["lease_until"] > self.clock():
            self._update(job_id, status="stopping", stop_reason=reason)
            return self.get(job_id)  # The active launcher checks this durable intent before/after start.
        container = self._inspect(job)
        # Never signal an unknown/missing container or claim it has stopped.
        if container is None:
            self._update(job_id, status="lost", reason="container missing during cancellation")
            return self.get(job_id)
        self._update(job_id, status="stopping", stop_reason=reason)
        if container.get("State", {}).get("Status") == "created":
            self._update(job_id, status=reason, reason="container never started; no candidate process exists")
            return self.get(job_id)
        if container.get("State", {}).get("Status") not in {"exited", "dead"}:
            reference = self._container_id(container)
            # Fresh ownership check and fixed ID avoid signalling a foreign
            # container that replaced the deterministic name after inspection.
            container = self._inspect(job, reference=reference)
            if container is None or self._container_id(container) != reference:
                raise DockerError("owned container changed before stop")
            result = self._call(["container", "kill", "--signal", "KILL", "--", reference])
            container = self._inspect(job, reference=reference)
            if container is not None and self._container_id(container) != reference:
                raise DockerError("container identity changed during stop confirmation")
            if result.returncode and container and container.get("State", {}).get("Status") not in {"exited", "dead"}:
                self._update(job_id, reason="kill not confirmed; container may still be running")
                return self.get(job_id)
        if container is None:
            self._update(job_id, status="lost", reason="container disappeared while stopping")
            return self.get(job_id)
        return self._observe(job_id, container) if container.get("State", {}).get("Status") in {"exited", "dead"} else self.get(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] in self.TERMINAL:
            return job
        # Queued cancellation uses the same transaction as launch claiming.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute("UPDATE docker_jobs SET status='cancelled',reason='cancelled before launch' "
                                 "WHERE job_id=? AND status='queued'", (job_id,)).rowcount
        if updated:
            self._request_stop(job_id, "cancelled")
            return self.get(job_id)
        self._request_stop(job_id, "cancelled")
        try:
            return self._terminate(job_id, "cancelled")
        except OwnershipError:
            raise
        except (DockerError, CapabilityError) as error:
            self._update(job_id, status="stopping", stop_reason="cancelled", reason=str(error))
            return self.get(job_id)

    def recover(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            jobs = [row[0] for row in db.execute("SELECT job_id FROM docker_jobs ORDER BY created")]
        values = []
        for job_id in jobs:
            try:
                values.append(self.poll(job_id))
            except (DockerError, CapabilityError, ValueError, OSError) as error:
                # An unknown container must not prevent supervision of other jobs.
                values.append({**self.get(job_id), "reconciliation_error": f"{type(error).__name__}: {error}"})
        return values

    def serve(self, *, interval: float = 1.0, stop: threading.Event | None = None,
              on_tick: Callable[[list[dict[str, Any]]], None] | None = None) -> None:
        """Operator-supervised monitor; restart promptly if it exits.

        A disconnected client does not stop this process. Run the separate
        stop-only ``reclab_agent.watchdog`` process to supervise deadlines even
        when this launcher is suspended/offline. Both supervisors still depend
        on a reachable, responsive Docker daemon.
        """
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or not 0.1 <= interval <= 10:
            raise ValueError("supervision interval must be between 0.1 and 10 seconds")
        self.capabilities()
        stop = stop or threading.Event()
        while not stop.is_set():
            results = self.recover()
            if on_tick:
                on_tick(results)
            stop.wait(interval)

    def _manifest(self, job_id: str) -> dict[str, Any]:
        root, files, total = self._output(job_id), {}, 0
        for parent, directories, names in os.walk(root, followlinks=False):
            for name in directories:
                _directory(Path(parent) / name)
            for name in names:
                path = Path(parent) / name
                relative = path.relative_to(root).as_posix()
                confined_path(root, relative)
                raw = _regular(path, max_bytes=self.policy.max_output_bytes)
                total += len(raw)
                if total > self.policy.max_output_bytes or len(files) >= 1024:
                    raise ValueError("output collection budget exceeded")
                files[relative] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        return {"files": files, "bytes": total, "sha256": content_hash(files)}

    def _verify_receipt(self, job: dict[str, Any]) -> None:
        if job["receipt"] is None or self._manifest(job["job_id"]) != job["receipt"]:
            raise ValueError("completed output no longer matches its frozen receipt")

    @_serialized_job(busy="raise")
    def collect_predictions(self, job_id: str) -> dict[str, list[int]]:
        job = self.poll(job_id)
        if job["status"] != "completed" or job["spec"]["phase"] != "predict":
            raise ValueError("only completed prediction jobs have collectable rankings")
        if job["policy_hash"] != self.policy.identity():
            raise ValueError("restore original output/security policy before collecting")
        self._verify_receipt(job)
        raw = _regular(confined_path(self._output(job_id), "predictions.json"), max_bytes=min(32_000_000, self.policy.max_output_bytes))
        def pairs(values):
            result = {}
            for key, value in values:
                if key in result:
                    raise ValueError("duplicate JSON key in predictions")
                result[key] = value
            return result
        try:
            value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        except (RecursionError, UnicodeError) as error:
            raise ValueError("prediction JSON nesting or encoding is invalid") from error
        if (not isinstance(value, dict) or not value or any(not isinstance(user, str) or not user
                or not isinstance(row, list) or len(row) > 1000 or any(type(item) is not int or item <= 0 for item in row)
                for user, row in value.items())):
            raise ValueError("predictions must map users to bounded positive-integer rankings")
        # Catalog membership, coverage, duplicate ranks and metrics belong to the trusted grader.
        return value

    @_serialized_job(busy="raise")
    def export_artifacts(self, job_id: str, destination: Path) -> dict[str, Any]:
        """Export receipt-verified outputs as inert bytes, never load model code."""
        from reclab_agent.infrastructure.files import atomic_write
        job = self.poll(job_id)
        if job["status"] != "completed" or job["policy_hash"] != self.policy.identity():
            raise ValueError("only completed jobs under the original policy can export artifacts")
        self._verify_receipt(job)
        copied = {}
        for name, metadata in job["receipt"]["files"].items():
            if Path(name).suffix not in {".json", ".jsonl", ".pt", ".pth"}:
                continue
            raw = _regular(confined_path(self._output(job_id), name), max_bytes=self.policy.max_output_bytes)
            if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
                raise ValueError("artifact changed during export")
            atomic_write(confined_path(destination, name), raw)
            copied[name] = metadata
        return copied

    def logs(self, job_id: str) -> str:
        job = self.get(job_id)
        if self._inspect(job) is None:
            return ""
        result = self._call(["container", "logs", "--tail", "200", "--", job["container"]])
        if result.returncode:
            raise DockerError("failed to collect container logs")
        return (result.stdout + result.stderr)[-64_000:]
