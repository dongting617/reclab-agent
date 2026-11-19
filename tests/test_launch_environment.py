"""Literal environment loading and SSH path adaptation; no keys or network."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reclab_agent.research import launch_environment as module


class LaunchEnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="launch-env-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = SimpleNamespace(model_env="TEST_MODEL_VAR", api_key_env="TEST_API_VAR")
        self.env = self.root / "agent.env"

    def load(self, text):
        self.env.write_text(text, encoding="utf-8")
        return module.load_agent_environment(self.env, self.config)

    def test_comments_export_quotes_and_existing_environment(self):
        with patch.dict(os.environ, {"TEST_API_VAR": "existing", "TEST_MODEL_VAR": "old"}):
            state = self.load("# local settings\nexport TEST_MODEL_VAR='model id'\nOPENAI_BASE_URL=\"https://example.invalid/v1\"\n")
            self.assertEqual(os.environ["TEST_MODEL_VAR"], "model id")
            self.assertEqual(os.environ["TEST_API_VAR"], "existing")
            self.assertEqual(state, {"OPENAI_BASE_URL": True, "TEST_API_VAR": True, "TEST_MODEL_VAR": True})
            self.assertTrue(all(type(value) is bool for value in state.values()))

    def test_explicit_values_override_and_empty_value_clears(self):
        with patch.dict(os.environ, {"TEST_API_VAR": "previous", "TEST_MODEL_VAR": "previous"}):
            state = self.load("TEST_API_VAR=fixture-value\nTEST_MODEL_VAR=\n")
            self.assertEqual(os.environ["TEST_API_VAR"], "fixture-value")
            self.assertEqual(os.environ["TEST_MODEL_VAR"], "")
            self.assertFalse(state["TEST_MODEL_VAR"])

    def test_shell_expressions_remain_literal(self):
        with patch.dict(os.environ, {}, clear=False), patch.object(module.subprocess, "run") as run:
            self.load("TEST_API_VAR='$(id)'\nTEST_MODEL_VAR=$HOME\n")
            self.assertEqual(os.environ["TEST_API_VAR"], "$(id)")
            self.assertEqual(os.environ["TEST_MODEL_VAR"], "$HOME")
            run.assert_not_called()

    def test_invalid_file_is_atomic_and_errors_do_not_include_values(self):
        invalid = ["TEST_API_VAR=new-private-value\nTEST_API_VAR=duplicate-secret\n",
                   "TEST_API_VAR=new-private-value\nPATH=/unsafe\n",
                   "TEST_API_VAR=new-private-value\nTEST_MODEL_VAR=\"unclosed-secret\n",
                   "TEST_API_VAR=new-private-value\nTEST_MODEL_VAR=value with spaces\n",
                   "TEST_API_VAR=new-private-value\nnot-an-assignment\n",
                   "TEST_API_VAR=new-private-value\nTEST_MODEL_VAR=bad\x00value\n"]
        for text in invalid:
            with self.subTest(text=text), patch.dict(os.environ, {"TEST_API_VAR": "existing"}):
                with self.assertRaises(ValueError) as raised:
                    self.load(text)
                self.assertEqual(os.environ["TEST_API_VAR"], "existing")
                self.assertNotIn("private-value", str(raised.exception))
                self.assertNotIn("secret", str(raised.exception))

    def test_utf8_bom_and_empty_file_preserve_existing_values(self):
        with patch.dict(os.environ, {"TEST_API_VAR": "existing"}):
            self.env.write_text("\ufeff# no overrides\n", encoding="utf-8")
            state = module.load_agent_environment(self.env, self.config)
            self.assertTrue(state["TEST_API_VAR"])
            self.assertEqual(os.environ["TEST_API_VAR"], "existing")


class LaunchRemoteToolsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="launch-ssh-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.connection = {"host": "example.invalid", "port": 22, "user": "root",
                           "key_path": "identity", "remote_root": "/gpu/task",
                           "python_executable": "/python", "known_hosts_path": "known_hosts"}
        self.path = self.root / "connection.json"
        self.config = SimpleNamespace(execution_backend="ssh_tmux", execution_connection_path=self.path,
                                      output_directory=self.root / "output", dataset_directory=self.root / "data",
                                      task=SimpleNamespace(budget=SimpleNamespace(max_job_seconds=600)))

    def write_connection(self, **overrides):
        self.path.write_text(json.dumps({**self.connection, **overrides}))

    def test_docker_has_no_remote_tools(self):
        self.config.execution_backend = "docker"
        with patch.object(module, "RemoteTools") as factory:
            self.assertIsNone(module.build_remote_tools(self.config))
            factory.assert_not_called()

    def test_native_tools_bind_local_roots_without_ssh_or_key_reads(self):
        self.write_connection()
        with patch.object(module, "RemoteTools") as factory, patch.object(module.subprocess, "run") as run:
            result = module.build_remote_tools(self.config)
            self.assertIs(result, factory.return_value)
            connection = factory.call_args.args[0]
            self.assertEqual(connection.key_path, self.root / "identity")
            self.assertEqual(factory.call_args.kwargs["max_timeout_seconds"], 600)
            self.assertEqual(factory.call_args.kwargs["upload_roots"], {
                "source": self.root / "output/snapshots", "public": self.root / "data/public"})
            run.assert_not_called()

    def test_windows_mode_requires_wsl_and_installed_executables(self):
        self.write_connection()
        with patch.object(module, "_is_wsl", return_value=False), self.assertRaisesRegex(ValueError, "WSL"):
            module.build_remote_tools(self.config, windows_ssh=True)
        with patch.object(module, "_is_wsl", return_value=True), patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(ValueError, "OpenSSH"):
                module.build_remote_tools(self.config, windows_ssh=True)

    def test_windows_connection_maps_drive_paths_without_reading_keys(self):
        self.write_connection(key_path="C:/Users/Example/.ssh/key", known_hosts_path="D:\\SSH Files\\known_hosts")
        with patch.object(module, "_is_wsl", return_value=True), patch.object(Path, "is_file", return_value=True), \
                patch.object(module, "RemoteTools") as factory:
            module.build_remote_tools(self.config, windows_ssh=True)
            connection = factory.call_args.args[0]
            self.assertEqual(connection.key_path.as_posix(), "/mnt/c/Users/Example/.ssh/key")
            self.assertEqual(connection.known_hosts_path.as_posix(), "/mnt/d/SSH Files/known_hosts")
            self.assertIs(factory.call_args.kwargs["command_runner"], module._windows_command_runner)

    def test_windows_configuration_remains_strict(self):
        self.write_connection(key_path="C:relative", known_hosts_path="C:/known_hosts")
        with patch.object(module, "_is_wsl", return_value=True), patch.object(Path, "is_file", return_value=True):
            with self.assertRaisesRegex(ValueError, "absolute drive"):
                module.build_remote_tools(self.config, windows_ssh=True)
            self.path.write_text('{"host":"a","host":"b"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.build_remote_tools(self.config, windows_ssh=True)

    def test_bridge_maps_only_local_paths_and_preserves_payload(self):
        argv = ["scp", "-i", "/mnt/c/Users/Example/key", "-o",
                "UserKnownHostsFile=/mnt/c/Users/Example/known_hosts", "--",
                "/mnt/e/Model Files/payload", "root@example.invalid:/gpu/input"]
        with patch.object(module.subprocess, "run") as run:
            module._windows_command_runner(argv, input=b"literal input", timeout=12)
            mapped = run.call_args.args[0]
            self.assertEqual(mapped[0], "/mnt/c/Windows/System32/OpenSSH/scp.exe")
            self.assertEqual(mapped[2], "C:\\Users\\Example\\key")
            self.assertEqual(mapped[4], "UserKnownHostsFile=C:\\Users\\Example\\known_hosts")
            self.assertEqual(mapped[-2], "E:\\Model Files\\payload")
            self.assertEqual(mapped[-1], argv[-1])
            self.assertEqual(run.call_args.kwargs, {"input": b"literal input", "timeout": 12})
        self.assertEqual(argv[0], "scp")

    def test_bridge_rejects_other_commands_and_unmounted_paths(self):
        with self.assertRaises(ValueError):
            module._windows_command_runner(["sh", "-c", "id"])
        for path in ("/tmp/key", "/mnt/c/../key", "/mnt/c/Users/./key"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                module._mounted_windows_path(path)


if __name__ == "__main__":
    unittest.main()
