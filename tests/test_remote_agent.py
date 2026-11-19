"""Standalone SSH helper protocol, transfer, and process-supervision checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from reclab_agent import remote_agent as remote


class RemoteAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="remote-helper-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "train.py"
        self.script.write_text("print('done')\n", encoding="utf-8")

    def request(self, **changes):
        payload = {"job_id": "job_1", "argv": [sys.executable, "-B", "train.py"],
                   "cwd": ".", "env": {"RECLAB_SEED": "7"}, "timeout_seconds": 10}
        payload.update(changes)
        return payload

    def call(self, operation, **fields):
        return remote.dispatch({"op": operation, "root": str(self.root), **fields})

    def test_confinement_rejects_cross_platform_traversal_and_reserved_writes(self):
        for relative in ("../other", "a/../b", "a//b", "/etc/passwd", "C:/outside", "a\\b", ".", ""):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                remote._path(self.root, relative)
        for relative in (".control/remote_agent.py", "jobs/job_1/status.json"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.call("mkdir", path=relative)
        with self.assertRaises(ValueError):
            remote._root("relative/path")

    def test_symlink_root_and_path_components_are_rejected(self):
        link = self.root / "linked"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        for function in (lambda: remote._root(str(link)), lambda: remote._path(self.root, "linked/train.py")):
            with self.assertRaises(ValueError):
                function()

    def test_request_accepts_explicit_python_script_and_filters_environment(self):
        result = remote._request(self.root, self.request())
        self.assertEqual(result["argv"], [sys.executable, "-B", str(self.script)])
        self.assertEqual(result["cwd"], str(self.root))
        self.assertEqual(result["env"], {"RECLAB_SEED": "7"})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "SSH_AUTH_SOCK": "/secret"}):
            clean = remote._environment({"RECLAB_SEED": "7"})
        self.assertNotIn("OPENAI_API_KEY", clean)
        self.assertNotIn("SSH_AUTH_SOCK", clean)
        self.assertEqual(clean["RECLAB_SEED"], "7")

    def test_request_rejects_shell_module_code_external_scripts_and_unsafe_env(self):
        requests = [self.request(argv=[sys.executable, "-c", "print(1)"]),
                    self.request(argv=[sys.executable, "-m", "subprocess"]),
                    self.request(argv=["sh", "train.py"]),
                    self.request(argv=[sys.executable, "../train.py"]),
                    self.request(argv=[sys.executable, str(self.root.parent / "outside.py")]),
                    self.request(env={"PYTHONPATH": "elsewhere"}),
                    self.request(env={"OPENAI_API_KEY": "secret"}),
                    self.request(job_id="../job"), self.request(timeout_seconds=float("nan"))]
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ValueError):
                remote._request(self.root, request)

    def test_upload_commit_checks_hash_and_is_idempotent(self):
        content = b"model code\n"
        description = {"path": "source/model.py", "size": len(content),
                       "sha256": hashlib.sha256(content).hexdigest()}
        prepared = self.call("prepare_upload", **description)
        stage = Path(prepared["staging_path"])
        self.assertEqual(stage.parent, self.root / ".control/transfers")
        stage.write_bytes(b"incorrect")
        with self.assertRaises(ValueError):
            self.call("commit_upload", staging_path=str(stage), **description)
        self.assertFalse((self.root / "source/model.py").exists())
        stage.write_bytes(content)
        committed = self.call("commit_upload", staging_path=str(stage), **description)
        self.assertEqual(committed["sha256"], description["sha256"])
        self.assertEqual(self.call("commit_upload", staging_path=str(stage), **description), committed)
        self.assertEqual((self.root / "source/model.py").read_bytes(), content)
        self.call("release_transfer", staging_path=str(stage))
        self.assertFalse(stage.with_suffix(".json").exists())

    def test_download_is_bounded_snapshot_and_cannot_release_arbitrary_files(self):
        content = b"result"
        (self.root / "result.json").write_bytes(content)
        with self.assertRaises(ValueError):
            self.call("prepare_download", path="result.json", max_bytes=2)
        prepared = self.call("prepare_download", path="result.json", max_bytes=100)
        stage = Path(prepared["staging_path"])
        (self.root / "result.json").write_bytes(b"changed")
        self.assertEqual(stage.read_bytes(), content)
        self.assertEqual(prepared["sha256"], hashlib.sha256(content).hexdigest())
        with self.assertRaises(ValueError):
            self.call("release_transfer", staging_path=str(self.script))
        self.call("release_transfer", staging_path=str(stage))
        self.assertFalse(stage.exists())

    def test_stat_rejects_cap_and_hard_links(self):
        with self.assertRaises(ValueError):
            self.call("stat", path="train.py", max_bytes=1)
        hard = self.root / "hard.py"
        os.link(self.script, hard)
        with self.assertRaises(ValueError):
            self.call("stat", path="hard.py")

    def test_tail_is_bounded_and_returns_last_lines(self):
        directory = self.root / "jobs/job_1"
        directory.mkdir(parents=True)
        (directory / "job.log").write_text("a\nb\nc\n", encoding="utf-8")
        self.assertEqual(self.call("tail", job_id="job_1", lines=2)["text"], "b\nc")
        (directory / "job.log").write_bytes(b"a" * (remote.MAX_TAIL_BYTES * 2))
        result = self.call("tail", job_id="job_1")
        self.assertLessEqual(result["bytes"], remote.MAX_TAIL_BYTES)
        self.assertTrue(result["truncated"])

    def test_control_cli_emits_machine_readable_error(self):
        result = subprocess.run([sys.executable, "-B", remote.__file__, "control"],
                                input=b"not json", capture_output=True)
        self.assertEqual(result.returncode, 1)
        envelope = json.loads(result.stdout)
        self.assertFalse(envelope["ok"])
        self.assertEqual(envelope["error"]["code"], "INVALID_ARGUMENTS")

    def test_hello_fingerprint_contains_only_stable_runtime_details(self):
        def command(argv):
            output = "12.8\n" if argv[0] == sys.executable else "fixed version/device\n"
            return {"exit_code": 0, "stdout": output, "stderr": ""}
        with patch.object(remote.importlib.metadata, "version", return_value="test-version"), \
                patch.object(remote, "_command", side_effect=command):
            first, second = self.call("hello"), self.call("hello")
        self.assertEqual(first, second)
        self.assertEqual(first["torch_cuda"], "12.8")
        self.assertEqual(first["helper_sha256"], hashlib.sha256(Path(remote.__file__).read_bytes()).hexdigest())
        self.assertNotIn("pid", first)
        self.assertNotIn("created_at", first)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux supervision")
    def test_run_deduplicates_same_job_id_and_refuses_changed_request(self):
        completed = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(remote.subprocess, "run", return_value=completed) as launch:
            first = self.call("run", **self.request())
            second = self.call("run", **self.request())
            self.assertEqual(first, second)
            self.assertEqual(launch.call_count, 1)
            self.assertIn("new-session", launch.call_args.args[0])
            with self.assertRaises(ValueError):
                self.call("run", **self.request(env={"RECLAB_SEED": "8"}))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux supervision")
    def test_missing_tmux_session_is_lost_never_success(self):
        directory = remote._job(self.root, "job_1", create=True)
        remote._atomic(directory / "status.json", {"status": "running", "session": "missing",
                                                  "exit_code": None, "stop_confirmed": False})
        with patch.object(remote, "_tmux_alive", return_value=False):
            result = self.call("status", job_id="job_1")
        self.assertEqual(result["status"], "lost")
        self.assertIsNone(result["exit_code"])
        self.assertFalse(result["stop_confirmed"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux supervision")
    def test_execute_records_success_failure_timeout_and_filters_credentials(self):
        scripts = [("ok", "import os\nprint(os.environ.get('OPENAI_API_KEY', 'clean'))\n", 5, "completed", 0),
                   ("bad", "raise SystemExit(3)\n", 5, "failed", 3),
                   ("slow", "import time\ntime.sleep(20)\n", .1, "timed_out", None)]
        for job_id, code, timeout, expected, exit_code in scripts:
            with self.subTest(job_id=job_id):
                self.script.write_text(code, encoding="utf-8")
                directory = remote._job(self.root, job_id, create=True)
                request = remote._request(self.root, self.request(job_id=job_id, timeout_seconds=timeout))
                remote._atomic(directory / "request.json", request)
                remote._atomic(directory / "status.json", {"job_id": job_id, "status": "starting"})
                with patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
                    remote._execute(str(self.root), job_id)
                result = remote._json(directory / "status.json")
                self.assertEqual(result["status"], expected)
                self.assertTrue(result["stop_confirmed"])
                if exit_code is not None:
                    self.assertEqual(result["exit_code"], exit_code)
                if job_id == "ok":
                    self.assertEqual((directory / "job.log").read_text().strip(), "clean")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux supervision")
    def test_pid_reuse_identity_blocks_group_signal(self):
        with patch.object(remote, "_identity", return_value="new"), patch.object(remote.os, "killpg") as kill:
            self.assertFalse(remote._signal_group({"pid": 111, "process_start": "old"}, 15))
            kill.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux supervision")
    def test_timeout_kills_term_resistant_process_group(self):
        child_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
        self.script.write_text("import signal,subprocess,sys,time\n"
                               "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                               f"subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
                               "time.sleep(60)\n", encoding="utf-8")
        directory = remote._job(self.root, "group", create=True)
        remote._atomic(directory / "request.json", remote._request(
            self.root, self.request(job_id="group", timeout_seconds=.3)))
        remote._atomic(directory / "status.json", {"status": "starting"})
        terminate = remote._terminate
        with patch.object(remote, "_terminate", side_effect=lambda p, identity: terminate(p, identity, grace=.1)):
            remote._execute(str(self.root), "group")
        result = remote._json(directory / "status.json")
        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["exit_code"], -9)
        self.assertTrue(result["stop_confirmed"])
        self.assertFalse(remote._group_active(result["pid"]))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux supervision")
    def test_cancelled_before_wrapper_claim_never_launches_target(self):
        directory = remote._job(self.root, "early_cancel", create=True)
        remote._atomic(directory / "request.json", remote._request(self.root, self.request(job_id="early_cancel")))
        remote._atomic(directory / "status.json", {"status": "starting"})
        remote._atomic(directory / "cancel.json", {"requested_at": time.time()})
        with patch.object(remote.subprocess, "Popen") as launch:
            remote._execute(str(self.root), "early_cancel")
        launch.assert_not_called()
        result = remote._json(directory / "status.json")
        self.assertEqual(result["status"], "cancelled")
        self.assertTrue(result["stop_confirmed"])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux supervision")
    def test_stop_uses_wrapper_acknowledgement_and_persists_cancelled(self):
        self.script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
        directory = remote._job(self.root, "cancel", create=True)
        remote._atomic(directory / "request.json", remote._request(self.root, self.request(job_id="cancel")))
        remote._atomic(directory / "status.json", {"status": "starting", "session": "unused"})
        process = subprocess.Popen([sys.executable, "-B", remote.__file__, "_execute", str(self.root), "cancel"])
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if remote._json(directory / "status.json").get("status") == "running":
                    break
                time.sleep(.05)
            else:
                self.fail("supervisor did not start")
            result = self.call("stop", job_id="cancel")
            self.assertEqual(result["status"], "cancelled")
            self.assertTrue(result["stop_confirmed"])
            self.assertEqual(process.wait(timeout=5), 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
