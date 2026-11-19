"""Probe contract tests; these do not claim a real Linux permission boundary."""
import contextlib
import errno
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_dataset_permissions.py"
SPEC = importlib.util.spec_from_file_location("permission_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class PermissionProbeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.public = self.root / "public"
        self.public.mkdir()
        self.write("manifest.json", {"dataset_id": "test-fixture", "test_labels_exposed": False})
        self.write("train.json", {"u": [1, 2]})
        self.write("valid.json", {"u": {"history": [1, 2], "target": 3}})

    def write(self, name, value):
        (self.public / name).write_text(json.dumps(value), encoding="utf-8")

    def test_exact_public_files_are_hashed_without_content_in_report(self):
        before = {p.name: p.read_bytes() for p in self.public.iterdir()}
        result = probe.validate_public(self.public)
        self.assertEqual(set(result), probe.PUBLIC_FILES)
        self.assertTrue(all(len(value) == 64 for value in result.values()))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.public.iterdir()})

    def test_extra_file_and_nested_directory_are_rejected(self):
        extra = self.public / "test.json"
        extra.write_text("{}", encoding="utf-8")
        with self.assertRaises(probe.ProbeError):
            probe.validate_public(self.public)
        extra.unlink()
        extra.mkdir()
        with self.assertRaises(probe.ProbeError):
            probe.validate_public(self.public)

    def test_hidden_file_is_also_rejected(self):
        (self.public / ".secret").write_text("hidden", encoding="utf-8")
        with self.assertRaises(probe.ProbeError):
            probe.validate_public(self.public)

    def test_test_keys_are_rejected_recursively_and_case_insensitively(self):
        for key in probe.FORBIDDEN_KEYS:
            with self.subTest(key=key):
                self.write("manifest.json", {"nested": [{key.upper(): 999}]})
                with self.assertRaises(probe.ProbeError):
                    probe.validate_public(self.public)

    def test_non_object_or_invalid_json_is_rejected(self):
        for value in ([], 1, "text", None):
            self.write("manifest.json", value)
            with self.assertRaises(probe.ProbeError):
                probe.validate_public(self.public)
        (self.public / "manifest.json").write_text("not-json", encoding="utf-8")
        with self.assertRaises(ValueError):
            probe.validate_public(self.public)

    def test_false_label_exposure_flag_cannot_be_replaced_by_true_or_zero(self):
        for flag in (True, 0, None, "false"):
            self.write("manifest.json", {"test_labels_exposed": flag})
            with self.assertRaises(probe.ProbeError):
                probe.validate_public(self.public)

    def test_relative_and_parent_traversal_paths_are_rejected(self):
        with self.assertRaises(probe.ProbeError):
            probe.validate_public(Path("public"))
        with self.assertRaises(probe.ProbeError):
            probe.validate_public(self.public / ".." / "public")

    def test_input_reparse_point_is_rejected_without_following(self):
        class FakeStat:
            st_mode = 0o100644
            st_file_attributes = 0x400
        with patch.object(Path, "lstat", return_value=FakeStat()):
            with self.assertRaises(probe.ProbeError):
                probe._no_links(self.public)

    def test_unsupported_platform_fails_before_reading_inputs(self):
        with patch.object(probe.sys, "platform", "win32"), patch.object(probe, "validate_public") as validate:
            with self.assertRaises(probe.ProbeError):
                probe.run_probe(self.public, self.root / "private.json", self.root / "original.json")
            validate.assert_not_called()

    def test_missing_private_file_is_not_treated_as_an_existing_denied_file(self):
        with self.assertRaises(FileNotFoundError):
            probe._private_file(self.root / "does-not-exist.json")

    def test_permission_error_records_errno_without_secret_content(self):
        with patch.object(probe.os, "open", side_effect=PermissionError(errno.EACCES, "secret-label-value")):
            result = probe._attempt(Path("/root/private/test.json"))
        self.assertEqual(result, {"allowed": False, "errno": errno.EACCES, "errno_name": "EACCES"})
        self.assertNotIn("secret-label-value", json.dumps(result))

    def check_report(self, private_errno):
        calls = ([{"allowed": True, "errno": None}] * 3
                 + [{"allowed": False, "errno": errno.EACCES}] * 3
                 + [{"allowed": False, "errno": private_errno}] * 4)
        with patch.object(probe, "_drop_privileges", return_value={"Uid": "65534"}), patch.object(probe, "_attempt", side_effect=calls):
            return probe._fixed_checks(Path("/tmp/stage"), Path("/root/private.json"), Path("/root/original.json"))

    def test_all_named_accesses_must_be_actually_permission_denied(self):
        self.assertEqual(self.check_report(errno.EACCES)["status"], "passed")
        self.assertEqual(self.check_report(errno.ENOENT)["status"], "failed")
        self.assertEqual(self.check_report(errno.EIO)["status"], "failed")

    def test_unexpected_private_access_is_a_failure(self):
        calls = ([{"allowed": True, "errno": None}] * 3
                 + [{"allowed": False, "errno": errno.EACCES}] * 3
                 + [{"allowed": True, "errno": None}] * 4)
        with patch.object(probe, "_drop_privileges", return_value={}), patch.object(probe, "_attempt", side_effect=calls):
            result = probe._fixed_checks(Path("/tmp/stage"), Path("/root/private"), Path("/root/source"))
        self.assertEqual(result["status"], "failed")

    def test_write_probe_never_requests_truncation(self):
        with patch.object(probe.os, "open", side_effect=PermissionError(errno.EACCES, "denied")) as opening:
            probe._attempt(Path("/root/file.json"), write=True)
        self.assertEqual(opening.call_args.args[1], probe.os.O_WRONLY)

    def test_cli_failure_is_nonzero_and_explicitly_not_a_sandbox(self):
        with patch.object(probe, "run_probe", side_effect=probe.ProbeError("Linux required")), contextlib.redirect_stdout(io.StringIO()) as output:
            code = probe.main(["--public-dir", str(self.public), "--private-test-path", "/root/private.json",
                               "--original-source-path", "/root/original.json"])
        result = json.loads(output.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["sandbox_established"])
        self.assertFalse(result["docker_verified"])
        self.assertIn("week two", " ".join(result["limits"]))

    def test_cli_writes_new_report_only_and_rejects_command_argument(self):
        report = self.root / "report.json"
        argv = ["--public-dir", str(self.public), "--private-test-path", "/root/private.json",
                "--original-source-path", "/root/original.json", "--output", str(report)]
        with patch.object(probe, "run_probe", side_effect=probe.ProbeError("Linux required")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(probe.main(argv), 1)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["status"], "failed")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.main(argv)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            probe.main(argv[:-2] + ["--command", "echo arbitrary-code"])


if __name__ == "__main__":
    unittest.main()
