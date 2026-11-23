"""Administrator-started, stop-only Docker deadline supervisor.

Run independently of the Worker launcher. It never creates, starts, removes or
executes containers, and it never runs candidate code on the host. A persistent
stop intent is checked on every later tick, including after a job is terminal.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sqlite3
import stat
import threading
import time
from typing import Any, Callable
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.execution.worker import (
    CapabilityError,
    CommandResult,
    DockerError,
    DockerWorker,
    JobSpec,
    WorkerPolicy,
    _directory,
)
from reclab_agent.infrastructure.files import confined_path
from reclab_agent.execution.config import load_policy


class DockerDeadlineWatchdog:
    """Stop owned jobs without waiting for a suspended launcher's OS lock.

    The job lock still fences create/start and terminal bookkeeping. Docker KILL
    deliberately does not acquire it: a SIGSTOP'ed launcher must not prevent the
    independent supervisor from stopping an already running container. All
    stop intents remain in the private database to catch a late create/start.
    """

    def __init__(self, root: str | Path, policy: WorkerPolicy, *,
                 runner: Callable[..., CommandResult] | None = None,
                 clock: Callable[[], float] = time.time):
        root = _directory(root)
        database = confined_path(root, "jobs.sqlite3")
        info = database.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("watchdog requires the existing private Worker database")
        # An accidental new/empty database is not a running Worker installation.
        db = sqlite3.connect(database)
        try:
            if db.execute("SELECT value FROM worker_meta WHERE key='owner'").fetchone() is None:
                raise ValueError("Worker owner identity is missing")
        finally:
            db.close()
        self.worker = DockerWorker(root, policy, runner=runner, clock=clock,
                                   initialize=False, database_timeout=0.05)
        self.clock = clock

    def _validate_job(self, job: dict[str, Any]) -> None:
        key = job["job_key"]
        if (not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key)
                or job["job_id"] != f"job-{key[:40]}"
                or job["container"] != f"reclab-{key[:40]}"
                or job["policy_hash"] != self.worker.policy.identity()
                or key != self.worker._key(JobSpec.from_dict(job["spec"]))):
            raise ValueError("restore the registered Worker policy and exact durable job identity")

    def _record(self, job_id: str, state: str, *, confirmed: bool, detail: str) -> str | None:
        now = self.clock()
        try:
            with self.worker._connect() as db:
                db.execute("UPDATE docker_stop_intents SET last_checked=?,container_state=?,confirmed=?,detail=? "
                           "WHERE job_id=?", (now, state, now if confirmed else None, detail, job_id))
        except sqlite3.Error as error:
            # Accounting failure is pending, not evidence of stopping, and must
            # not suppress an independent stop of the next known job.
            return f"SQLite accounting pending: {type(error).__name__}: {error}"
        return None

    def _owned_container(self, job: dict[str, Any], *, reference: str | None = None) -> dict[str, Any] | None:
        container = self.worker._inspect(job, reference=reference)
        if container is not None:
            # Signal the immutable ID, not a name that could have been replaced
            # since inspection. No arbitrary Docker references cross the CLI.
            container_id = self.worker._container_id(container)
            if reference and container_id != reference:
                raise DockerError("container identity changed during stop confirmation")
        return container

    def _stop(self, job: dict[str, Any], reason: str, *, existing_intent: bool = False) -> dict[str, Any]:
        job_id = job["job_id"]
        self._validate_job(job)
        container = self._owned_container(job)
        state = container.get("State", {}).get("Status") if container else "missing"
        journal_reason = self.worker._journal_reason(job_id)
        existing_intent = existing_intent or bool(job["stop_reason"] or journal_reason)
        if not existing_intent and state in {"exited", "dead"}:
            if type(container["State"].get("ExitCode")) is not int:
                raise DockerError("container exit code is missing")
            finished_at = container["State"].get("FinishedAt")
            if finished_at is not None:
                try:
                    finished = datetime.fromisoformat(finished_at.replace("Z", "+00:00")).timestamp()
                except (ValueError, AttributeError, TypeError, OverflowError) as error:
                    raise DockerError("container finish timestamp is invalid") from error
                if not math.isfinite(finished) or finished <= 0:
                    raise DockerError("container finish timestamp is invalid")
            else:
                finished = None
            if finished is None or finished <= job["deadline"]:
                # Inspection after the deadline is not execution after it.
                # Only Worker validates outputs and assigns completed/failed.
                return {"job_id": job_id, "status": job["status"], "stop_reason": None,
                        "container_state": state, "stop_confirmed": True, "launcher_busy": False,
                        "detail": "already exited before deadline, or finish time unknown; Worker must reconcile outputs"}
        reason = journal_reason or job["stop_reason"] or reason
        pending = []
        # Private filesystem tombstone is independent of SQLite's write lock.
        # Persist it before signalling so a resumed launcher also sees the stop.
        reason = self.worker._journal_stop(job_id, reason)
        try:
            reason = self.worker._request_stop(job_id, reason)
        except sqlite3.Error as error:
            pending.append(f"SQLite stop intent pending: {type(error).__name__}: {error}")
        if container and state not in {"created", "exited", "dead"}:
            reference = container["Id"]
            # A nonzero/timeout response does not prove failure or success. Only
            # the fresh ownership-checked inspection below can confirm exit.
            try:
                self.worker._call(["container", "kill", "--signal", "KILL", "--", reference])
            except (DockerError, CapabilityError):
                pass
            container = self._owned_container(job, reference=reference)
            state = container.get("State", {}).get("Status") if container else "missing"

        confirmed = state in {"exited", "dead"} and type(container["State"].get("ExitCode")) is int
        detail = ("exit confirmed; persistent stop supervision remains active" if confirmed else
                  "stop not confirmed; created/missing/active containers require continued supervision")
        launcher_busy = False
        # This lock affects terminal bookkeeping only, never the KILL above.
        with self.worker._job_lock(job_id) as acquired:
            launcher_busy = not acquired
            if acquired and confirmed:
                latest = self._owned_container(job, reference=container["Id"])
                latest_state = latest.get("State", {}) if latest else {}
                confirmed = latest_state.get("Status") in {"exited", "dead"} and type(latest_state.get("ExitCode")) is int
                if confirmed:
                    try:
                        self.worker._update(job_id, status=reason, stop_reason=reason,
                                            exit_code=latest_state["ExitCode"], reason="independent watchdog confirmed exit")
                    except sqlite3.Error as error:
                        pending.append(f"SQLite terminal accounting pending: {type(error).__name__}: {error}")
                else:
                    state = latest_state.get("Status", "missing")
                    detail = "state changed before bookkeeping; persistent stop supervision remains active"
        if launcher_busy:
            detail += "; launcher lock is busy, terminal bookkeeping deferred"
        record_error = self._record(job_id, str(state), confirmed=confirmed, detail=detail)
        if record_error:
            pending.append(record_error)
        try:
            status = self.worker.get(job_id)["status"]
        except sqlite3.Error as error:
            status = job["status"]
            pending.append(f"SQLite status refresh pending: {type(error).__name__}: {error}")
        detail += ("; " + "; ".join(pending)) if pending else ""
        return {"job_id": job_id, "status": status, "stop_reason": reason,
                "container_state": state, "stop_confirmed": confirmed, "launcher_busy": launcher_busy,
                "accounting_pending": bool(pending), "detail": detail}

    def tick(self) -> list[dict[str, Any]]:
        """One bounded stop-only scan. Never schedules or starts queued jobs."""
        with self.worker._connect() as db:
            jobs = {row["job_id"]: (row["stop_reason"] or row["intent_reason"] or "timed_out",
                                    bool(row["stop_reason"] or row["intent_reason"]))
                    for row in db.execute(
                        "SELECT j.job_id,j.stop_reason,i.reason AS intent_reason FROM docker_jobs j "
                        "LEFT JOIN docker_stop_intents i ON i.job_id=j.job_id "
                        "WHERE i.job_id IS NOT NULL OR j.stop_reason IS NOT NULL OR "
                        "(j.deadline IS NOT NULL AND j.deadline<=? AND "
                        "j.status NOT IN ('completed','failed','timed_out','cancelled','lost')) ORDER BY j.created",
                        (self.clock(),))}
        journal = confined_path(self.worker.root, "stop-journal")
        if journal.exists():
            for path in journal.iterdir():
                match = re.fullmatch(r"(job-[a-f0-9]{40})\.(cancelled|timed_out)", path.name)
                if match:
                    jobs.setdefault(match[1], (match[2], True))
        results = []
        for job_id, (reason, existing_intent) in jobs.items():
            try:
                results.append(self._stop(self.worker.get(job_id), reason, existing_intent=existing_intent))
            except (DockerError, CapabilityError, ValueError, KeyError, OSError, sqlite3.Error) as error:
                # An unknown container/failed daemon must not suppress stops of
                # other known jobs. The error never counts as confirmed exit.
                detail = f"{type(error).__name__}: {error}"
                record_error = self._record(job_id, "uncertain", confirmed=False, detail=detail)
                if record_error:
                    detail += "; " + record_error
                results.append({"job_id": job_id, "stop_confirmed": False, "detail": detail})
        return results

    def serve(self, *, interval: float = 1.0, stop: threading.Event | None = None,
              on_tick: Callable[[list[dict[str, Any]]], None] | None = None) -> None:
        """Run until explicitly stopped; operator supervises this separate process."""
        if (isinstance(interval, bool) or not isinstance(interval, (int, float))
                or not math.isfinite(interval) or not 0.1 <= interval <= 10):
            raise ValueError("supervision interval must be between 0.1 and 10 seconds")
        stop = stop or threading.Event()
        while not stop.is_set():
            results = self.tick()
            if on_tick:
                on_tick(results)
            stop.wait(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="existing private Worker state on the Docker host")
    parser.add_argument("--policy", required=True, help="same administrator-approved immutable Worker policy")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("once")
    serve = commands.add_parser("serve")
    serve.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        watchdog = DockerDeadlineWatchdog(args.state, load_policy(args.policy))
        if args.command == "once":
            result = watchdog.tick()
            print(canonical_json({"ok": True, "result": result}))
        else:
            watchdog.serve(interval=args.interval,
                           on_tick=lambda rows: print(canonical_json({"ok": True, "result": rows}), flush=True) if rows else None)
        return 0
    except (CapabilityError, DockerError, ValueError, KeyError, OSError, sqlite3.Error) as error:
        print(canonical_json({"ok": False, "error": type(error).__name__, "detail": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
