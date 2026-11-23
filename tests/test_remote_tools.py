"""SSH transport contracts using fakes only; these tests never open a connection."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from reclab_agent.tools.remote import RemoteTools
from reclab_agent.execution.ssh import SSHConnection


class FakeSSH:
    def __init__(self):
        self.calls = []
        self.requests = []
        self.files = {}
        self.staging = {}
        self.prepared = {}
        self.scp_payloads = []
        self.hooks = {}
        self.failure = None
        self.counter = 0

    @staticmethod
    def metadata(raw):
        return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if self.failure is not None:
            if isinstance(self.failure, Exception):
                raise self.failure
            return self.failure
        if argv[0] == "scp":
            first, second = argv[-2:]
            download = first.startswith("root@")
            endpoint, local = (first, second) if download else (second, first)
            remote = shlex.split(endpoint.split(":", 1)[1])[0]
            if download:
                Path(local).write_bytes(self.staging[remote])
            else:
                raw = Path(local).read_bytes()
                self.scp_payloads.append(raw)
                if remote.endswith("/remote_agent.py"):
                    self.files[remote] = raw
                else:
                    self.staging[remote] = raw
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if kwargs.get("input") is None:
            command = shlex.split(argv[-1])
            if "-c" in command:
                operation, root, name, _ = command[-4:]
                stage = root + "/.control/" + name
                if operation == "prepare":
                    self.staging[stage] = b""
                elif operation == "install":
                    self.files[root + "/.control/remote_agent.py"] = self.staging.pop(stage)
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        request = json.loads(kwargs["input"])
        self.requests.append(request)
        op = request["op"]
        if op in self.hooks:
            result = self.hooks[op](request)
            if isinstance(result, subprocess.CompletedProcess):
                return result
        else:
            result = self.dispatch(request)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"ok": True, "result": result}).encode(), b"")

    def dispatch(self, request):
        op, root = request["op"], request["root"]
        if op == "hello":
            return {"helper_sha256": self.metadata(self.files[root + "/.control/remote_agent.py"])["sha256"],
                    "root": root, "python": "/gpu/bin/python", "protocol_version": 1}
        if op == "prepare_upload":
            self.counter += 1
            stage = root + "/.control/transfers/" + f"{self.counter:032x}"
            self.prepared[stage] = request
            self.staging[stage] = b""
            return {"staging_path": stage, "size": request["size"], "sha256": request["sha256"]}
        if op == "commit_upload":
            raw = self.staging[request["staging_path"]]
            metadata = self.metadata(raw)
            assert metadata == {"size": request["size"], "sha256": request["sha256"]}
            self.files[request["path"]] = raw
            return {"path": root + "/" + request["path"], **metadata}
        if op == "prepare_download":
            raw = self.files[request["path"]]
            self.counter += 1
            stage = root + "/.control/transfers/" + f"{self.counter:032x}"
            self.staging[stage] = raw
            return {"staging_path": stage, **self.metadata(raw)}
        if op == "release_transfer":
            self.staging.pop(request["staging_path"], None)
            return {"released": True}
        if op == "stat":
            return {"path": root + "/" + request["path"], **self.metadata(self.files[request["path"]])}
        if op == "gpu_status":
            return {"returncode": 0, "stdout": "0, test-gpu, 0, 24000, 0\n", "stderr": ""}
        if op == "tail":
            return {"job_id": request["job_id"], "text": "diagnostic only\n"}
        if op == "stop":
            return {"job_id": request["job_id"], "status": "cancelled", "exit_code": -15, "stop_confirmed": True}
        return {"job_id": request["job_id"], "status": "starting", "exit_code": None, "stop_confirmed": False}


class RemoteToolsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="remote-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "approved-source"
        self.source.mkdir()
        self.key = self.root / "identity"
        self.key.write_text("not a real key", encoding="utf-8")
        self.known = self.root / "known_hosts"
        self.known.write_text("test-host", encoding="utf-8")
        self.connection = SSHConnection("gpu.example.com", 42052, "root", self.key,
                                        "/gpu/task", "/gpu/bin/python", self.known)
        self.runner = FakeSSH()
        self.tools = RemoteTools(self.connection, upload_roots={"source": self.source},
                                 download_root=self.root / "downloads", command_runner=self.runner,
                                 max_transfer_bytes=4096, max_timeout_seconds=600)

    def run_request(self):
        return self.tools.run_remote("baseline_1", ["/gpu/bin/python", "-B", "train.py", "--config", "config.json"],
                                     "source", 120, seed=7)

    def test_seven_strict_tools_hide_connection_configuration(self):
        specs = self.tools.specs
        self.assertEqual({s.name for s in specs}, {
            "upload", "download", "run_remote", "job_status", "tail_log", "gpu_status", "stop_job"})
        for spec in specs:
            schema = spec.input_schema
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            self.assertFalse({"host", "port", "key_path", "remote_root", "command", "env"} & schema["properties"].keys())
        self.assertEqual(self.tools.invoke("bootstrap")["error"]["code"], "UNKNOWN_TOOL")
        self.assertFalse(self.tools.invoke("gpu_status", {"host": "other.example.com"})["ok"])

    def test_arguments_travel_as_json_and_host_key_checking_is_mandatory(self):
        payload = "$(touch /tmp/side-effect); 'quoted value'"
        result = self.tools.run_remote("stable_1", ["/gpu/bin/python", "-u", "train.py", "--label", payload],
                                       "source", 30, 9)
        self.assertEqual(result["status"], "starting")
        request = self.runner.requests[-1]
        self.assertEqual(request["env"], {"RECLAB_SEED": "9"})
        self.assertEqual(request["argv"][-1], payload)
        command, options = self.runner.calls[-1]
        self.assertNotIn(payload, command[-1])
        self.assertEqual(shlex.split(command[-1]), ["/gpu/bin/python", "-B", "/gpu/task/.control/remote_agent.py", "control"])
        self.assertIn("BatchMode=yes", command)
        self.assertIn("StrictHostKeyChecking=yes", command)
        self.assertIn("IdentitiesOnly=yes", command)
        self.assertIn("ForwardAgent=no", command)
        self.assertIn("UserKnownHostsFile=" + str(self.known), command)
        self.assertNotIn("shell", options)
        self.assertEqual(command[command.index("-p") + 1], "42052")

    def test_rejects_shells_python_inline_code_and_unconfined_scripts_before_ssh(self):
        invalid = [
            ["bash", "-c", "train.py"], ["/gpu/bin/python", "-c", "print(1)"],
            ["/gpu/bin/python", "-m", "train"], ["/gpu/bin/python", "/outside/train.py"],
            ["/gpu/bin/python", "../train.py"], ["/gpu/bin/python", "/gpu/task/.control/remote_agent.py"],
            ["/gpu/bin/python", "-B"], ["/gpu/bin/python", "train.py", "\x00"],
        ]
        for argv in invalid:
            with self.subTest(argv=argv):
                response = self.tools.invoke("run_remote", {"job_id": "j1", "argv": argv,
                        "cwd": ".", "timeout_seconds": 30, "seed": 1})
                self.assertEqual(response["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(self.runner.calls, [])
        for override in ({"job_id": "../bad"}, {"timeout_seconds": True}, {"timeout_seconds": 601},
                         {"seed": True}, {"cwd": "a/../b"}):
            arguments = {"job_id": "j1", "argv": ["/gpu/bin/python", "train.py"],
                         "cwd": ".", "timeout_seconds": 30, "seed": 1, **override}
            self.assertFalse(self.tools.invoke("run_remote", arguments)["ok"])
        self.assertEqual(self.runner.calls, [])

    def test_upload_uses_frozen_bytes_and_hash_checked_staging(self):
        source = self.source / "train.py"
        source.write_bytes(b"print('reviewed')\n")
        expected = source.read_bytes()

        def prepare(request):
            source.write_bytes(b"later change\n")
            return self.runner.dispatch(request)

        self.runner.hooks["prepare_upload"] = prepare
        result = self.tools.upload("source", "train.py", "source/train.py")
        self.assertEqual(self.runner.files["source/train.py"], expected)
        self.assertEqual(self.runner.scp_payloads, [expected])
        self.assertEqual(result["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual([r["op"] for r in self.runner.requests], ["prepare_upload", "commit_upload", "release_transfer"])
        self.assertFalse(self.runner.staging)
        scp = next(argv for argv, _ in self.runner.calls if argv[0] == "scp")
        self.assertIn("-O", scp)
        self.assertIn("StrictHostKeyChecking=yes", scp)

    def test_upload_rejects_unknown_roots_traversal_control_paths_and_oversized_files(self):
        (self.source / "train.py").write_text("pass", encoding="utf-8")
        cases = [
            ("private", "train.py", "source/train.py"), ("source", "../identity", "source/train.py"),
            ("source", "train.py", "../../outside"), ("source", "train.py", ".control/remote_agent.py"),
            ("source", "train.py", "C:/outside"), ("source", "train.py", "source\\train.py"),
        ]
        for root, source, destination in cases:
            with self.subTest(source=source, destination=destination):
                result = self.tools.invoke("upload", {"source_root": root, "source": source, "destination": destination})
                self.assertFalse(result["ok"])
        (self.source / "big").write_bytes(b"a" * 4097)
        self.assertFalse(self.tools.invoke("upload", {"source_root": "source", "source": "big", "destination": "data/big"})["ok"])
        self.assertEqual(self.runner.calls, [])
        result = self.tools.invoke("upload", {"source_root": "source", "source": "missing", "destination": "data/missing"})
        self.assertEqual(result["error"]["code"], "LOCAL_IO_ERROR")

    def test_download_snapshots_and_verifies_before_atomic_destination_replace(self):
        self.runner.files["output/predictions.json"] = b'{"user": [1, 2]}'
        target = self.tools.download_root / "results/predictions.json"
        result = self.tools.download("output/predictions.json", "results/predictions.json")
        self.assertEqual(target.read_bytes(), self.runner.files["output/predictions.json"])
        self.assertEqual(result["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertEqual([r["op"] for r in self.runner.requests], ["prepare_download", "release_transfer"])
        self.assertFalse(self.runner.staging)

        def incorrect_hash(request):
            return {**self.runner.dispatch(request), "sha256": "0" * 64}

        self.runner.hooks["prepare_download"] = incorrect_hash
        before = target.read_bytes()
        response = self.tools.invoke("download", {"source": "output/predictions.json", "destination": "results/predictions.json"})
        self.assertEqual(response["error"]["code"], "HASH_MISMATCH")
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(self.runner.staging)

    def test_download_limits_and_staging_scope_are_checked_before_scp(self):
        self.runner.files["output/result"] = b"x"
        for override in ({"size": 4097}, {"size": True}, {"sha256": "invalid"},
                         {"staging_path": "/gpu/elsewhere/result"},
                         {"staging_path": "/gpu/task/source/train.py"}):
            with self.subTest(override=override):
                self.runner.hooks["prepare_download"] = lambda request: {**self.runner.dispatch(request), **override}
                before = len([c for c, _ in self.runner.calls if c[0] == "scp"])
                response = self.tools.invoke("download", {"source": "output/result", "destination": "result"})
                self.assertFalse(response["ok"])
                self.assertEqual(len([c for c, _ in self.runner.calls if c[0] == "scp"]), before)
                self.assertFalse((self.tools.download_root / "result").exists())
        count = len(self.runner.calls)
        for source, destination in ((".control/secret", "result"), ("output/result", "../identity")):
            self.assertFalse(self.tools.invoke("download", {"source": source, "destination": destination})["ok"])
        self.assertEqual(len(self.runner.calls), count)

    def test_remote_errors_and_transport_uncertainty_keep_distinct_envelopes(self):
        self.runner.hooks["status"] = lambda request: subprocess.CompletedProcess([], 1,
            json.dumps({"ok": False, "error": {"code": "NOT_FOUND", "message": "job does not exist"}}).encode(), b"")
        response = self.tools.invoke("job_status", {"job_id": "missing"})
        self.assertEqual(response["error"]["code"], "NOT_FOUND")
        self.runner.failure = subprocess.CompletedProcess([], 255, b"", b"Permission denied (publickey)")
        self.assertEqual(self.tools.invoke("gpu_status")["error"]["code"], "SSH_FAILED")
        self.runner.failure = subprocess.TimeoutExpired(["ssh"], 60)
        self.assertEqual(self.tools.invoke("job_status", {"job_id": "stable"})["error"]["code"], "SSH_TIMEOUT")
        self.runner.failure = FileNotFoundError("ssh not installed")
        self.assertEqual(self.tools.invoke("gpu_status")["error"]["code"], "SSH_UNAVAILABLE")

    def test_bad_control_json_and_oversized_responses_are_rejected(self):
        bad = [b"not json", b'{"ok":true,"result":[],"ok":false}', b'{"ok":true,"result":[]}',
               b'{"ok":1,"result":{}}', b'{"ok":true,"result":{"bad":NaN}}', b"x" * 1_048_577]
        for raw in bad:
            with self.subTest(raw=raw[:60]):
                self.runner.failure = subprocess.CompletedProcess([], 0, raw, b"")
                self.assertEqual(self.tools.invoke("gpu_status")["error"]["code"], "REMOTE_PROTOCOL")

    def test_status_tail_stop_and_host_only_file_stat_use_same_registered_connection(self):
        self.runner.files["source/train.py"] = b"pass\n"
        self.run_request()
        self.assertIsNone(self.tools.job_status("baseline_1")["exit_code"])
        self.assertIn("diagnostic", self.tools.tail_log("baseline_1", 200)["text"])
        self.assertTrue(self.tools.stop_job("baseline_1")["stop_confirmed"])
        self.assertEqual(self.tools.file_stat("source/train.py")["size"], 5)
        self.assertNotIn("file_stat", {spec.name for spec in self.tools.specs})
        self.assertFalse(self.tools.invoke("tail_log", {"job_id": "baseline_1", "lines": 1001})["ok"])
        self.assertEqual({r["root"] for r in self.runner.requests}, {"/gpu/task"})

    def test_bootstrap_snapshots_reviewed_helper_and_checks_digest(self):
        helper = self.root / "helper.py"
        helper.write_bytes(b"# reviewed helper\n")
        result = self.tools.bootstrap(helper)
        self.assertEqual(result["helper_sha256"], hashlib.sha256(helper.read_bytes()).hexdigest())
        self.assertEqual(self.runner.scp_payloads, [helper.read_bytes()])
        self.runner.hooks["hello"] = lambda request: {"helper_sha256": "f" * 64}
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.tools.bootstrap(helper)

    def test_registration_accepts_future_snapshot_root_but_rejects_symlink_components(self):
        future = self.root / "future/snapshots"
        tools = RemoteTools(self.connection, upload_roots={"source": future},
                            download_root=self.root / "downloads", command_runner=self.runner)
        self.assertEqual(tools.upload_roots["source"], future)
        self.assertTrue(future.is_dir())
        link = self.source / "linked"
        try:
            link.symlink_to(self.key)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is unavailable")
        result = self.tools.invoke("upload", {"source_root": "source", "source": "linked", "destination": "source/linked"})
        self.assertFalse(result["ok"])
        self.assertEqual(self.runner.calls, [])


class SSHConnectionConfigurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ssh-config-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "connection.json"
        self.value = {"host": "gpu.example.com", "port": 42052, "user": "root", "key_path": "keys/id_ed25519",
                      "known_hosts_path": "keys/known_hosts", "remote_root": "/gpu/task", "python_executable": "/gpu/bin/python"}

    def test_config_local_paths_resolve_relative_to_config_not_process_cwd(self):
        self.path.write_text(json.dumps(self.value), encoding="utf-8")
        connection = SSHConnection.from_json(self.path)
        self.assertEqual(connection.key_path, self.root / "keys/id_ed25519")
        self.assertEqual(connection.known_hosts_path, self.root / "keys/known_hosts")
        self.assertEqual(connection.helper_path, "/gpu/task/.control/remote_agent.py")

    def test_config_rejects_unknown_keys_duplicate_keys_and_unsafe_connection_values(self):
        for override in ({"password": "forbidden"}, {"port": True}, {"host": "-oProxyCommand=evil"},
                         {"host": "user@gpu.example.com"}, {"user": "root;evil"}, {"key_path": None},
                         {"remote_root": "/gpu/../task"}, {"remote_root": "/"}, {"python_executable": "python"}):
            with self.subTest(override=override):
                self.path.write_text(json.dumps({**self.value, **override}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    SSHConnection.from_json(self.path)
        raw = json.dumps(self.value)[:-1] + ',"host":"other.example.com"}'
        self.path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SSHConnection.from_json(self.path)


class LocalBootstrapSSH(FakeSSH):
    """Execute only captured trusted bootstrap snippets in a temporary directory."""

    def __init__(self):
        super().__init__()
        self.corrupt_stage = False

    def __call__(self, argv, **kwargs):
        if argv[0] == "ssh" and kwargs.get("input") is None:
            self.calls.append((list(argv), kwargs))
            command = shlex.split(argv[-1])
            if command[1:3] != ["-B", "-c"]:
                raise AssertionError("only the fixed bootstrap Python is executed by this fake")
            return subprocess.run([sys.executable, *command[1:]], **kwargs)
        if argv[0] == "scp":
            self.calls.append((list(argv), kwargs))
            source, endpoint = argv[-2:]
            destination = Path(shlex.split(endpoint.split(":", 1)[1])[0])
            raw = Path(source).read_bytes()
            if self.corrupt_stage:
                raw += b"changed in transfer"
            destination.write_bytes(raw)
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return super().__call__(argv, **kwargs)

    def dispatch(self, request):
        if request["op"] == "hello":
            helper = Path(request["root"]) / ".control/remote_agent.py"
            return {"helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest()}
        return super().dispatch(request)


@unittest.skipUnless(os.name == "posix", "bootstrap snippets target a Linux GPU")
class BootstrapFilesystemTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bootstrap-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "gpu-task"
        self.helper = self.root / "reviewed.py"
        self.helper.write_bytes(b"# reviewed helper\n")
        self.runner = LocalBootstrapSSH()
        connection = SSHConnection("gpu.example.com", 42052, "root", self.root / "unused-key",
                                   self.remote.as_posix(), "/registered/python")
        self.tools = RemoteTools(connection, upload_roots={}, download_root=self.root / "download",
                                 command_runner=self.runner)

    def test_bootstrap_checks_and_atomically_installs_exact_bytes(self):
        self.tools.bootstrap(self.helper)
        target = self.remote / ".control/remote_agent.py"
        self.assertEqual(target.read_bytes(), self.helper.read_bytes())
        self.assertEqual(list(target.parent.glob("helper-*.upload")), [])
        self.assertFalse(any(call[0][0] not in {"ssh", "scp"} for call in self.runner.calls))

    def test_existing_directory_and_helper_symlinks_are_rejected_before_scp(self):
        outside = self.root / "outside"
        outside.mkdir()
        sentinel = outside / "remote_agent.py"
        sentinel.write_bytes(b"leave this file alone")
        for location in (self.remote, self.remote / ".control", self.remote / ".control/remote_agent.py"):
            with self.subTest(location=location):
                if location != self.remote:
                    location.parent.mkdir(parents=True, exist_ok=True)
                location.symlink_to(sentinel if location.name == "remote_agent.py" else outside,
                                    target_is_directory=location.name != "remote_agent.py")
                count = len([c for c, _ in self.runner.calls if c[0] == "scp"])
                with self.assertRaisesRegex(ValueError, "SSH transport exited"):
                    self.tools.bootstrap(self.helper)
                self.assertEqual(len([c for c, _ in self.runner.calls if c[0] == "scp"]), count)
                self.assertEqual(sentinel.read_bytes(), b"leave this file alone")
                location.unlink()

    def test_corrupted_transfer_preserves_preexisting_reviewed_helper(self):
        control = self.remote / ".control"
        control.mkdir(parents=True)
        target = control / "remote_agent.py"
        target.write_bytes(b"previous reviewed helper")
        self.runner.corrupt_stage = True
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.tools.bootstrap(self.helper)
        self.assertEqual(target.read_bytes(), b"previous reviewed helper")


if __name__ == "__main__":
    unittest.main()
