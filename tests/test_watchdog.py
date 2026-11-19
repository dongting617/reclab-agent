"""Independent stop-only supervision; fake Docker is not isolation evidence."""
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from reclab_agent.datasets import create_synthetic_bundle
from reclab_agent.execution import (DockerError, DockerWorker, InputTree,
                                    JobSpec, ResourceLimits, WorkerPolicy)
from reclab_agent.watchdog import DockerDeadlineWatchdog, main
from tests.support.docker import IMAGE, WatchdogFakeDocker, _hold_job_lock


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        for relative, value in {"src/train.py": "raise RuntimeError('never run on host')\n",
                                "src/predict.py": "raise RuntimeError('never run on host')\n",
                                "configs/default.json": "{}"}.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        self.bundle = create_synthetic_bundle(self.root / "data")
        self.policy = WorkerPolicy(IMAGE, {"baseline": InputTree.freeze(self.source, "source")},
                                   {"tiny": InputTree.freeze(self.bundle.public, "public_data")})
        self.fake, self.now = WatchdogFakeDocker(), 1000.0
        self.state = self.root / "state"
        self.worker = DockerWorker(self.state, self.policy, runner=self.fake, clock=lambda: self.now)
        self.spec = JobSpec("baseline", "tiny", limits=ResourceLimits(seconds=2))
        self.watchdog = DockerDeadlineWatchdog(self.state, self.policy, runner=self.fake, clock=lambda: self.now)

    def overdue(self):
        job = self.worker.submit(self.spec)
        self.now += 3
        return job

    def intent(self, job_id):
        with self.worker._connect() as db:
            return dict(db.execute("SELECT * FROM docker_stop_intents WHERE job_id=?", (job_id,)).fetchone())

    def test_does_not_create_database_or_schedule_queued_jobs(self):
        with self.assertRaises(ValueError):
            DockerDeadlineWatchdog(self.root / "missing", self.policy, runner=self.fake)
        self.worker.submit(self.spec)
        queued = self.worker.submit(replace(self.spec, seed=43))
        self.assertEqual(queued["status"], "queued")
        before = list(self.fake.calls)
        self.assertEqual(self.watchdog.tick(), [])
        self.assertEqual(self.fake.calls, before)

    def test_kills_overdue_by_immutable_owned_id_and_confirms(self):
        job = self.overdue()
        self.fake.calls.clear()
        result = self.watchdog.tick()[0]
        self.assertEqual(result["status"], "timed_out")
        self.assertTrue(result["stop_confirmed"])
        self.assertEqual(self.fake.calls[1], ["container", "kill", "--signal", "KILL", "--",
                                               self.fake.containers[job["container"]]["Id"]])
        self.assertTrue(all(call[:2] in (["container", "inspect"], ["container", "kill"]) for call in self.fake.calls))
        self.assertEqual(self.intent(job["job_id"])["confirmed"], self.now)

    def test_nonzero_kill_and_transport_loss_require_inspection_confirmation(self):
        job = self.overdue()
        self.fake.kill_confirmed = False
        result = self.watchdog.tick()[0]
        self.assertFalse(result["stop_confirmed"])
        self.assertEqual(result["status"], "stopping")
        self.assertIsNone(self.intent(job["job_id"])["confirmed"])
        original = self.fake
        def runner(args, timeout=10):
            result = original(args, timeout=timeout)
            if args[:2] == ["container", "kill"]:
                raise DockerError("connection lost after successful kill")
            return result
        self.fake.kill_confirmed = True
        self.watchdog.worker.runner = runner
        self.assertTrue(self.watchdog.tick()[0]["stop_confirmed"])
        self.assertEqual(self.worker.get(job["job_id"])["status"], "timed_out")

    def test_unknown_labels_or_missing_id_never_receive_signal(self):
        job = self.overdue()
        container = self.fake.containers[job["container"]]
        container["Config"]["Labels"]["io.reclab.worker"] = "unknown"
        result = self.watchdog.tick()[0]
        self.assertIn("OwnershipError", result["detail"])
        self.assertFalse(result["stop_confirmed"])
        self.assertEqual(self.fake.count("kill"), 0)
        container["Config"]["Labels"]["io.reclab.worker"] = self.worker.owner
        del container["Id"]
        self.assertIn("immutable ID", self.watchdog.tick()[0]["detail"])
        self.assertEqual(self.fake.count("kill"), 0)

    def test_disconnected_inspection_does_not_count_as_stopped(self):
        job = self.overdue()
        self.fake.inspect_disconnect = True
        result = self.watchdog.tick()[0]
        self.assertFalse(result["stop_confirmed"])
        self.assertIn("disconnect", result["detail"])
        # A failed first inspect cannot establish whether this job already
        # finished on time. Do not pre-emptively persist a timeout outcome.
        self.assertEqual(self.worker.get(job["job_id"])["status"], "running")
        self.assertIsNone(self.worker._journal_reason(job["job_id"]))
        self.assertEqual(self.fake.count("kill"), 0)

    def test_completion_before_deadline_is_not_overwritten_by_late_watchdog_observation(self):
        job = self.worker.submit(self.spec)
        (self.worker._output(job["job_id"]) / "model.json").write_text("{}", encoding="utf-8")
        self.fake.containers[job["container"]]["State"].update(
            Status="exited", ExitCode=0, FinishedAt=datetime.fromtimestamp(1001, timezone.utc).isoformat())
        self.now = 1003
        result = self.watchdog.tick()[0]
        self.assertIsNone(result["stop_reason"])
        self.assertEqual(self.fake.count("kill"), 0)
        self.assertIsNone(self.worker._journal_reason(job["job_id"]))
        self.assertEqual(self.worker.poll(job["job_id"])["status"], "completed")
        self.assertEqual(self.watchdog.tick(), [])

    def test_sqlite_paused_writer_does_not_prevent_owned_overdue_stop_or_watchdog_start(self):
        job = self.overdue()
        writer = sqlite3.connect(self.worker.database)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("UPDATE docker_jobs SET reason='uncommitted' WHERE job_id=?", (job["job_id"],))
            # Startup reads existing schema/owner only: no DDL or INSERT lock.
            restarted = DockerDeadlineWatchdog(self.state, self.policy, runner=self.fake, clock=lambda: self.now)
            result = restarted.tick()[0]
            self.assertTrue(result["stop_confirmed"])
            self.assertTrue(result["accounting_pending"])
            self.assertIn("SQLite", result["detail"])
            self.assertEqual(self.fake.count("kill"), 1)
            self.assertEqual(self.worker._journal_reason(job["job_id"]), "timed_out")
            # Repeated accounting failure neither crashes scan nor loses intent.
            self.assertTrue(restarted.tick()[0]["stop_confirmed"])
        finally:
            writer.rollback()
            writer.close()
        self.assertEqual(restarted.tick()[0]["status"], "timed_out")
        self.assertEqual(self.fake.count("kill"), 1)

    def test_malformed_container_state_isolated_from_other_jobs(self):
        first = self.worker.submit(self.spec)
        self.worker._update(first["job_id"], status="failed")
        self.worker._request_stop(first["job_id"], "cancelled")
        self.fake.containers[first["container"]]["State"] = None
        second = self.worker.submit(replace(self.spec, seed=43))
        self.now += 3
        results = {row["job_id"]: row for row in self.watchdog.tick()}
        self.assertIn("invalid container inspection", results[first["job_id"]]["detail"])
        self.assertFalse(results[first["job_id"]]["stop_confirmed"])
        self.assertTrue(results[second["job_id"]]["stop_confirmed"])
        self.assertEqual(self.fake.count("kill"), 1)

    def test_worker_cancellation_signals_old_id_when_name_replaced_by_foreign_container(self):
        job = self.worker.submit(self.spec)
        original_container = self.fake.containers[job["container"]]
        original_id = original_container["Id"]
        foreign = {"Id": "b" * 64, "Config": {"Labels": {"io.reclab.worker": "foreign"}},
                   "State": {"Status": "running", "ExitCode": 0}}
        original_runner = self.fake
        def replace_name(args, timeout=10):
            if args[:2] == ["container", "kill"]:
                self.fake.containers[job["container"]] = foreign
            return original_runner(args, timeout=timeout)
        self.worker.runner = replace_name
        result = self.worker.cancel(job["job_id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(foreign["State"]["Status"], "running")
        self.assertEqual(original_container["State"]["Status"], "exited")
        self.assertEqual([call[-1] for call in self.fake.calls if call[:2] == ["container", "kill"]], [original_id])

    def test_unknown_first_container_does_not_block_stopping_another_known_job(self):
        first = self.worker.submit(self.spec)
        self.worker._update(first["job_id"], status="failed")
        self.worker._request_stop(first["job_id"], "cancelled")
        self.fake.containers[first["container"]]["Config"]["Labels"]["io.reclab.worker"] = "unknown"
        second = self.worker.submit(replace(self.spec, seed=43))
        self.now += 3
        results = {row["job_id"]: row for row in self.watchdog.tick()}
        self.assertIn("OwnershipError", results[first["job_id"]]["detail"])
        self.assertFalse(results[first["job_id"]]["stop_confirmed"])
        self.assertTrue(results[second["job_id"]]["stop_confirmed"])
        self.assertEqual(results[second["job_id"]]["status"], "timed_out")
        self.assertEqual(self.fake.count("kill"), 1)

    def test_persistent_intent_catches_late_created_missing_and_restarted_terminal_job(self):
        job = self.overdue()
        container = self.fake.containers[job["container"]]
        container["State"]["Status"] = "created"
        first = self.watchdog.tick()[0]
        self.assertEqual(first["container_state"], "created")
        self.assertFalse(first["stop_confirmed"])
        self.assertEqual(first["status"], "stopping")
        del self.fake.containers[job["container"]]
        self.assertEqual(self.watchdog.tick()[0]["container_state"], "missing")
        self.assertFalse(self.watchdog.tick()[0]["stop_confirmed"])
        # A delayed old start finally arrives. New watcher process reads the same
        # durable intent and stops it without ever starting a container itself.
        self.fake.containers[job["container"]] = container
        container["State"]["Status"] = "running"
        restarted = DockerDeadlineWatchdog(self.state, self.policy, runner=self.fake, clock=lambda: self.now)
        self.assertEqual(restarted.tick()[0]["status"], "timed_out")
        container["State"]["Status"] = "running"
        self.assertTrue(restarted.tick()[0]["stop_confirmed"])
        self.assertEqual(self.fake.count("kill"), 2)
        self.assertEqual(self.fake.count("start"), 1)
        self.assertEqual(self.fake.count("create"), 1)

    def test_cancel_intent_precedes_deadline_and_first_reason_survives_restart(self):
        job = self.worker.submit(self.spec)
        self.fake.kill_confirmed = False
        self.worker.cancel(job["job_id"])
        self.fake.kill_confirmed = True
        self.now += 3
        self.assertEqual(self.watchdog.tick()[0]["status"], "cancelled")
        self.assertEqual(self.intent(job["job_id"])["reason"], "cancelled")
        self.worker._request_stop(job["job_id"], "timed_out")
        self.assertEqual(self.intent(job["job_id"])["reason"], "cancelled")

    def test_old_launcher_paused_between_check_and_start_is_still_supervised(self):
        paused, release = threading.Event(), threading.Event()
        original = self.fake
        errors, results = [], []
        def runner(args, timeout=10):
            if args[:2] == ["container", "start"]:
                paused.set()
                if not release.wait(10):
                    raise DockerError("test pause expired")
            return original(args, timeout=timeout)
        self.worker.runner = runner
        def launch():
            try:
                results.append(self.worker.submit(self.spec))
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=launch)
        thread.start()
        try:
            self.assertTrue(paused.wait(5))
            self.now += 61  # Even an expired lease cannot steal the OS lock.
            observed = self.watchdog.tick()[0]
            self.assertEqual(observed["container_state"], "created")
            self.assertTrue(observed["launcher_busy"])
            self.assertFalse(observed["stop_confirmed"])
            self.assertEqual(self.fake.count("start"), 0)
        finally:
            release.set()
            thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors, errors)
        # The compliant resumed Worker sees the tombstone after start and stops;
        # a stale/non-compliant launcher is also caught on every later tick.
        self.assertEqual(results[0]["status"], "timed_out")
        self.assertTrue(self.watchdog.tick()[0]["stop_confirmed"])
        self.assertEqual(self.fake.count("create"), 1)
        self.assertEqual(self.fake.count("start"), 1)

    def _independent_lock_case(self, *, sigstop=False):
        job = self.overdue()
        context = multiprocessing.get_context("spawn")
        ready, release = context.Event(), context.Event()
        process = context.Process(target=_hold_job_lock,
                                  args=(str(self.state), str(self.source), str(self.bundle.public),
                                        job["job_id"], ready, release))
        process.start()
        stopped = False
        try:
            self.assertTrue(ready.wait(10))
            if sigstop:
                os.kill(process.pid, signal.SIGSTOP)
                stopped = True
            before = self.fake.count("kill")
            result = self.watchdog.tick()[0]
            self.assertTrue(result["launcher_busy"])
            self.assertTrue(result["stop_confirmed"])
            self.assertEqual(result["status"], "stopping")
            self.assertEqual(self.fake.count("kill"), before + 1)
            self.assertEqual(self.fake.count("create"), 1)
            self.assertEqual(self.fake.count("start"), 1)
        finally:
            if stopped:
                os.kill(process.pid, signal.SIGCONT)
            release.set()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(self.watchdog.tick()[0]["status"], "timed_out")

    def test_real_cross_process_job_lock_cannot_block_independent_kill(self):
        self._independent_lock_case()

    @unittest.skipUnless(sys.platform == "linux", "SIGSTOP fault injection requires Linux")
    def test_sigstopped_launcher_still_cannot_block_independent_kill(self):
        self._independent_lock_case(sigstop=True)

    def test_invalid_policy_or_job_identity_never_signals(self):
        job = self.overdue()
        changed = DockerDeadlineWatchdog(self.state, replace(self.policy, max_concurrent_jobs=2),
                                         runner=self.fake, clock=lambda: self.now)
        self.assertIn("policy", changed.tick()[0]["detail"])
        self.assertEqual(self.fake.count("kill"), 0)
        with self.worker._connect() as db:
            db.execute("UPDATE docker_jobs SET container='unknown' WHERE job_id=?", (job["job_id"],))
        self.assertIn("identity", self.watchdog.tick()[0]["detail"])
        self.assertEqual(self.fake.count("kill"), 0)

    def test_serve_has_explicit_stop_and_validates_polling_interval(self):
        job = self.overdue()
        stop = threading.Event()
        observations = []
        def observed(rows):
            observations.extend(rows)
            stop.set()
        self.watchdog.serve(interval=0.1, stop=stop, on_tick=observed)
        self.assertTrue(observations[0]["stop_confirmed"])
        self.assertEqual(self.worker.get(job["job_id"])["status"], "timed_out")
        for invalid in (True, 0, -1, 11, float("nan"), "1"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.watchdog.serve(interval=invalid)

    def test_cli_only_exposes_once_and_serve_no_candidate_or_daemon_install(self):
        with patch("reclab_agent.watchdog.load_policy", return_value=self.policy), \
             patch("reclab_agent.watchdog.DockerDeadlineWatchdog", return_value=self.watchdog), \
             patch("builtins.print") as printed:
            self.assertEqual(main(["--state", str(self.state), "--policy", "approved.json", "once"]), 0)
            self.assertTrue(json.loads(printed.call_args.args[0])["ok"])
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            main(["--state", str(self.state), "--policy", "approved.json", "start", "--command", "sh"])


if __name__ == "__main__":
    unittest.main()
