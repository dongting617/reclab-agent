"""Small, host-configured SSH tools for a dedicated GPU and detached tmux jobs.

This is a transport, not a sandbox. Connection details and permitted local roots
are supplied by trusted host code, never as model tool arguments. The remote
helper accepts JSON on stdin; only its fixed invocation crosses a remote shell.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shlex
import subprocess
import tempfile
from typing import Any, Callable, Mapping
import uuid

from .files import atomic_write, confined_path, read_regular_file, validated_directory
from .tools.contracts import ToolError, ToolSpec
from .tools.registry import ToolRegistry


_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL_LIMIT = 1_048_576

# This program is trusted transport code, never supplied through a model tool.
# Validate every existing directory and file before SCP can touch the helper.
_BOOTSTRAP = """
import hashlib, os, pathlib, re, stat, sys
operation, root_value, name, expected = sys.argv[1:]
root = pathlib.Path(root_value)
if not root.is_absolute() or not re.fullmatch(r"helper-[0-9a-f]{32}\\.upload", name):
    raise ValueError("invalid bootstrap target")
directory = root / ".control"
for component in reversed((directory, *directory.parents)):
    try:
        info = component.lstat()
    except FileNotFoundError:
        if operation != "prepare":
            raise
        component.mkdir(mode=0o700)
        info = component.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError("bootstrap directory components must be ordinary directories")
target, stage = directory / "remote_agent.py", directory / name
def ordinary(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 1000000:
        raise ValueError("bootstrap file must be a bounded single-link regular file")
    return info
try:
    ordinary(target)
except FileNotFoundError:
    pass
if operation == "prepare":
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.close(descriptor)
elif operation == "install":
    info = ordinary(stage)
    descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as reader:
        opened = os.fstat(reader.fileno())
        if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("bootstrap staging identity changed")
        raw = reader.read(1000001)
    if len(raw) > 1000000 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bootstrap helper SHA-256 mismatch")
    os.replace(stage, target)
else:
    raise ValueError("unknown bootstrap operation")
"""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n"):
        raise ValueError(f"{label} must be nonempty text without control characters")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


def _absolute_posix(value: Any, label: str) -> str:
    value = _text(value, label)
    path = PurePosixPath(value)
    if (not path.is_absolute() or value.startswith("//") or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/")[1:])):
        raise ValueError(f"{label} must be a normalized absolute POSIX path")
    return str(path)


def _relative(value: Any, *, allow_root: bool = False) -> str:
    value = _text(value, "remote path")
    if allow_root and value == ".":
        return value
    path, windows = PurePosixPath(value), PureWindowsPath(value)
    if (path.is_absolute() or windows.drive or windows.root or "\\" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or path.parts[0] in {".control", ".reclab-upload"}):
        raise ValueError("remote path must remain within the task root and outside control directories")
    return str(path)


def _unique_json(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


@dataclass(frozen=True)
class SSHConnection:
    """Trusted SSH identity and one absolute remote task directory."""

    host: str
    port: int
    user: str
    key_path: Path
    remote_root: str
    python_executable: str
    known_hosts_path: Path | None = None

    def __post_init__(self) -> None:
        if (not isinstance(self.host, str)
                or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", self.host)):
            raise ValueError("host must be a literal DNS name or IPv4 address")
        _integer(self.port, "port", 1, 65535)
        if not isinstance(self.user, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", self.user):
            raise ValueError("invalid SSH user")
        object.__setattr__(self, "remote_root", _absolute_posix(self.remote_root, "remote_root"))
        if self.remote_root == "/":
            raise ValueError("remote_root cannot be the filesystem root")
        object.__setattr__(self, "python_executable", _absolute_posix(self.python_executable, "python_executable"))
        if self.key_path is None:
            raise ValueError("key_path must be a literal local path")
        for name in ("key_path", "known_hosts_path"):
            value = getattr(self, name)
            if value is not None:
                text = _text(os.fspath(value), name)
                if text.startswith("~"):
                    raise ValueError(f"{name} must be a literal local path")
                object.__setattr__(self, name, Path(os.path.abspath(text)))

    @property
    def helper_path(self) -> str:
        return str(PurePosixPath(self.remote_root) / ".control" / "remote_agent.py")

    @classmethod
    def from_json(cls, path: str | Path) -> SSHConnection:
        """Load only documented connection fields; local paths are config-relative."""
        path = Path(os.path.abspath(os.fspath(path)))
        value = json.loads(read_regular_file(path, max_bytes=65_536).decode("utf-8"),
                           object_pairs_hook=_unique_json, parse_constant=_reject_constant)
        required = {"host", "port", "user", "key_path", "remote_root", "python_executable"}
        if not isinstance(value, dict) or not required <= value.keys() or not value.keys() <= required | {"known_hosts_path"}:
            raise ValueError("SSH configuration must contain the documented connection fields only")
        for key in ("key_path", "known_hosts_path"):
            if key in value and value[key] is not None:
                text = _text(value[key], key)
                if os.name != "nt" and (PureWindowsPath(text).drive or "\\" in text):
                    raise ValueError(f"{key} uses a Windows path on a non-Windows host")
                candidate = Path(text)
                value[key] = candidate if candidate.is_absolute() else path.parent / candidate
        return cls(**value)


class RemoteTools:
    """Seven SSH primitives with bounded file transfer and stable job identities."""

    def __init__(self, connection: SSHConnection, *, upload_roots: Mapping[str, str | Path],
                 download_root: str | Path, command_runner: Callable[..., Any] = subprocess.run,
                 max_transfer_bytes: int = 512_000_000, max_timeout_seconds: int = 86_400):
        if not isinstance(connection, SSHConnection):
            raise TypeError("connection must be an SSHConnection")
        self.connection = connection
        self.max_transfer_bytes = _integer(max_transfer_bytes, "max_transfer_bytes", 1, 8 * 1024**3)
        self.max_timeout_seconds = _integer(max_timeout_seconds, "max_timeout_seconds", 1, 604_800)
        if not isinstance(upload_roots, Mapping):
            raise TypeError("upload_roots must be a mapping of registered names to directories")
        self.upload_roots: dict[str, Path] = {}
        for name, root in upload_roots.items():
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise ValueError("invalid upload root name")
            # Snapshot roots are commonly registered before the first freeze.
            self.upload_roots[name] = validated_directory(root, create=True)
        self.download_root = validated_directory(download_root, create=True)
        if not callable(command_runner):
            raise TypeError("command_runner must be callable")
        self._runner = command_runner
        self._tools = ToolRegistry()
        string = {"type": "string"}
        self._register("upload", "Upload one registered source or public-data file and verify its SHA-256.", {
            "source_root": {"type": "string", "enum": sorted(self.upload_roots)},
            "source": string, "destination": string,
        }, self.upload)
        self._register("download", "Download one bounded artifact and verify its SHA-256 before saving it.", {
            "source": string, "destination": string,
        }, self.download)
        self._register("run_remote", "Start or recover one stable tmux job; use job_status to observe its explicit exit code.", {
            "job_id": string, "argv": {"type": "array", "items": string, "minItems": 2},
            "cwd": string, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": self.max_timeout_seconds},
            "seed": {"type": "integer", "minimum": 0, "maximum": 2_147_483_647},
        }, self.run_remote)
        self._register("job_status", "Read a persisted job status; tmux absence alone does not mean success.",
                       {"job_id": string}, self.job_status)
        self._register("tail_log", "Read a bounded diagnostic tail of the job log.", {
            "job_id": string, "lines": {"type": "integer", "minimum": 1, "maximum": 1000},
        }, self.tail_log)
        self._register("gpu_status", "Read current GPU availability from nvidia-smi.", {}, self.gpu_status)
        self._register("stop_job", "Stop the matching job process group and retain its final status.",
                       {"job_id": string}, self.stop_job)

    def _register(self, name: str, description: str, properties: dict[str, Any], handler: Callable) -> None:
        def invoke(**arguments: Any) -> dict[str, Any]:
            try:
                return handler(**arguments)
            except OSError as error:
                raise ToolError("LOCAL_IO_ERROR", str(error)) from error

        self._tools.register_function(ToolSpec(name, description, {
            "type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False,
        }), invoke)

    @property
    def specs(self) -> list[ToolSpec]:
        return self._tools.specs

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._tools.invoke(name, arguments)

    def _options(self) -> list[str]:
        result = ["-i", os.fspath(self.connection.key_path), "-o", "BatchMode=yes",
                  "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
                  "-o", "ConnectTimeout=15", "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes"]
        if self.connection.known_hosts_path is not None:
            result += ["-o", f"UserKnownHostsFile={os.fspath(self.connection.known_hosts_path)}"]
        return result

    @property
    def _target(self) -> str:
        return f"{self.connection.user}@{self.connection.host}"

    def _ssh(self, command: str) -> list[str]:
        return ["ssh", "-p", str(self.connection.port), *self._options(), "--", self._target, command]

    def _execute(self, argv: list[str], *, input: bytes | None = None, timeout: int = 60,
                 helper_response: bool = False) -> bytes:
        try:
            completed = self._runner(argv, input=input, capture_output=True, check=False, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise ToolError("SSH_TIMEOUT", "SSH operation timed out; query the same job_id before retrying a job") from error
        except OSError as error:
            raise ToolError("SSH_UNAVAILABLE", f"Cannot start SSH transport: {error}") from error
        if completed.returncode != 0 and not (helper_response and completed.returncode != 255 and completed.stdout):
            detail = completed.stderr or b""
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise ToolError("SSH_FAILED", f"SSH transport exited with code {completed.returncode}: {detail[-4096:].strip()}")
        output = completed.stdout or b""
        return output.encode("utf-8") if isinstance(output, str) else output

    def _control(self, operation: str, **arguments: Any) -> dict[str, Any]:
        command = shlex.join([self.connection.python_executable, "-B", self.connection.helper_path, "control"])
        request = {"op": operation, "root": self.connection.remote_root, **arguments}
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(payload) > _CONTROL_LIMIT:
            raise ValueError("remote control request exceeds its size limit")
        raw = self._execute(self._ssh(command), input=payload, helper_response=True)
        if len(raw) > _CONTROL_LIMIT:
            raise ToolError("REMOTE_PROTOCOL", "Remote helper response exceeds the control response limit")
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json, parse_constant=_reject_constant)
        except (UnicodeError, ValueError) as error:
            raise ToolError("REMOTE_PROTOCOL", "Remote helper returned invalid JSON") from error
        if not isinstance(value, dict) or type(value.get("ok")) is not bool:
            raise ToolError("REMOTE_PROTOCOL", "Remote helper returned an invalid envelope")
        if value["ok"]:
            if not isinstance(value.get("result"), dict):
                raise ToolError("REMOTE_PROTOCOL", "Remote helper result must be an object")
            return value["result"]
        error = value.get("error")
        if not isinstance(error, dict) or not isinstance(error.get("message"), str):
            raise ToolError("REMOTE_PROTOCOL", "Remote helper returned an invalid error")
        code = error.get("code", "REMOTE_ERROR")
        if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", code):
            code = "REMOTE_ERROR"
        raise ToolError(code, error["message"][:4096])

    def _scp(self, local: Path, remote: str, *, download: bool = False) -> None:
        # Explicit legacy SCP mode gives shell quoting one documented meaning on
        # both old and new OpenSSH. Model paths never become shell command text.
        endpoint = f"{self._target}:{shlex.quote(remote)}"
        pair = [endpoint, os.fspath(local)] if download else [os.fspath(local), endpoint]
        command = ["scp", "-O", "-P", str(self.connection.port), *self._options(), "--", *pair]
        self._execute(command, timeout=300)

    def bootstrap(self, helper_path: str | Path | None = None) -> dict[str, Any]:
        """Host-only helper deployment; deliberately absent from the tool registry."""
        source = Path(helper_path) if helper_path is not None else Path(__file__).with_name("remote_agent.py")
        raw = read_regular_file(source, max_bytes=1_000_000)
        digest = hashlib.sha256(raw).hexdigest()
        name = "helper-" + uuid.uuid4().hex + ".upload"
        def bootstrap_command(operation: str) -> list[str]:
            command = shlex.join([self.connection.python_executable, "-B", "-c", _BOOTSTRAP,
                                  operation, self.connection.remote_root, name, digest])
            return self._ssh(command)

        self._execute(bootstrap_command("prepare"))
        staging = str(PurePosixPath(self.connection.remote_root) / ".control" / name)
        with tempfile.TemporaryDirectory(prefix=".ssh-helper-", dir=self.download_root) as temporary:
            snapshot = Path(temporary) / "remote_agent.py"
            snapshot.write_bytes(raw)
            self._scp(snapshot, staging)
        self._execute(bootstrap_command("install"))
        result = self._control("hello")
        if not hmac.compare_digest(str(result.get("helper_sha256", "")), digest):
            raise ToolError("HELPER_HASH_MISMATCH", "Deployed remote helper does not match the local reviewed helper")
        return result

    def _staging_path(self, value: Any) -> str:
        try:
            path = PurePosixPath(_absolute_posix(value, "staging_path"))
        except ValueError as error:
            raise ToolError("REMOTE_PROTOCOL", "Remote helper returned an invalid staging path") from error
        try:
            relative = path.relative_to(PurePosixPath(self.connection.remote_root))
        except ValueError as error:
            raise ToolError("REMOTE_PROTOCOL", "Remote staging path escaped the task root") from error
        parts = relative.parts
        if not (len(parts) == 3 and parts[:2] == (".control", "transfers")):
            raise ToolError("REMOTE_PROTOCOL", "Remote helper returned an unrecognized transfer staging path")
        if not re.fullmatch(r"[0-9a-f]{32}", parts[-1]):
            raise ToolError("REMOTE_PROTOCOL", "Remote helper returned an invalid transfer staging name")
        return str(path)

    def _release(self, staging: str) -> None:
        try:
            self._control("release_transfer", staging_path=staging)
        except ToolError:
            # A broken connection must not mask a transfer's primary result.
            # Staging names are unique and cannot be used as executable scripts.
            pass

    def _file_metadata(self, result: dict[str, Any]) -> tuple[int, str]:
        size, digest = result.get("size"), result.get("sha256")
        if (isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= self.max_transfer_bytes
                or not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
            raise ToolError("INVALID_ARTIFACT", "Remote file metadata is invalid or exceeds the transfer size limit")
        return size, digest

    def upload(self, source_root: str, source: str, destination: str) -> dict[str, Any]:
        if not isinstance(source_root, str) or source_root not in self.upload_roots:
            raise ValueError("source_root must name a host-registered upload directory")
        destination = _relative(destination)
        local = confined_path(self.upload_roots[source_root], source)
        raw = read_regular_file(local, max_bytes=self.max_transfer_bytes)
        digest = hashlib.sha256(raw).hexdigest()
        prepared = self._control("prepare_upload", path=destination, size=len(raw), sha256=digest)
        staging = self._staging_path(prepared.get("staging_path"))
        try:
            with tempfile.TemporaryDirectory(prefix=".ssh-upload-", dir=self.download_root) as temporary:
                snapshot = Path(temporary) / "payload"
                snapshot.write_bytes(raw)
                self._scp(snapshot, staging)
            result = self._control("commit_upload", path=destination, staging_path=staging, size=len(raw), sha256=digest)
            size, actual = self._file_metadata(result)
            if size != len(raw) or not hmac.compare_digest(actual, digest):
                raise ToolError("HASH_MISMATCH", "Remote upload did not match the source file")
            return {**result, "source_root": source_root, "source": source, "destination": destination}
        finally:
            self._release(staging)

    def download(self, source: str, destination: str) -> dict[str, Any]:
        source = _relative(source)
        target = confined_path(self.download_root, destination)
        prepared = self._control("prepare_download", path=source, max_bytes=self.max_transfer_bytes)
        size, digest = self._file_metadata(prepared)
        staging = self._staging_path(prepared.get("staging_path"))
        try:
            with tempfile.TemporaryDirectory(prefix=".ssh-download-", dir=self.download_root) as temporary:
                local = Path(temporary) / "payload"
                self._scp(local, staging, download=True)
                raw = read_regular_file(local, max_bytes=self.max_transfer_bytes)
        finally:
            self._release(staging)
        if len(raw) != size or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), digest):
            raise ToolError("HASH_MISMATCH", "Downloaded artifact changed or failed SHA-256 verification")
        target = confined_path(self.download_root, destination)
        atomic_write(target, raw)
        return {"source": source, "destination": destination, "size": size, "sha256": digest}

    def file_stat(self, path: str) -> dict[str, Any]:
        """Host-only metadata check for frozen remote inputs."""
        result = self._control("stat", path=_relative(path), max_bytes=self.max_transfer_bytes)
        self._file_metadata(result)
        return result

    def _job_id(self, job_id: Any) -> str:
        if not isinstance(job_id, str) or not _IDENTIFIER.fullmatch(job_id):
            raise ValueError("job_id must contain 1 to 80 letters, digits, underscores or hyphens")
        return job_id

    def run_remote(self, job_id: str, argv: list[str], cwd: str, timeout_seconds: int,
                   seed: int = 2026) -> dict[str, Any]:
        job_id = self._job_id(job_id)
        cwd = _relative(cwd, allow_root=True)
        _integer(timeout_seconds, "timeout_seconds", 1, self.max_timeout_seconds)
        _integer(seed, "seed", 0, 2_147_483_647)
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item or "\x00" in item for item in argv):
            raise ValueError("argv must be a nonempty array of nonempty strings")
        if argv[0] != self.connection.python_executable:
            raise ValueError("argv must use the registered remote Python executable")
        index = 1
        while index < len(argv) and argv[index] in {"-B", "-u"}:
            index += 1
        if index == len(argv) or argv[index].startswith("-"):
            raise ValueError("argv must execute a Python script, without -c or -m")
        script = argv[index]
        if PurePosixPath(script).is_absolute():
            try:
                script = str(PurePosixPath(script).relative_to(PurePosixPath(self.connection.remote_root)))
            except ValueError as error:
                raise ValueError("Python script must be within the remote task root") from error
        else:
            script = str(PurePosixPath(cwd) / script) if cwd != "." else script
        _relative(script)
        return self._control("run", job_id=job_id, argv=argv, cwd=cwd,
                             timeout_seconds=timeout_seconds, env={"RECLAB_SEED": str(seed)})

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._control("status", job_id=self._job_id(job_id))

    def tail_log(self, job_id: str, lines: int = 100) -> dict[str, Any]:
        _integer(lines, "lines", 1, 1000)
        return self._control("tail", job_id=self._job_id(job_id), lines=lines)

    def gpu_status(self) -> dict[str, Any]:
        return self._control("gpu_status")

    def stop_job(self, job_id: str) -> dict[str, Any]:
        return self._control("stop", job_id=self._job_id(job_id))
