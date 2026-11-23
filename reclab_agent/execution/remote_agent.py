"""Standalone JSON-over-stdin helper for a single SSH/tmux experiment root.

This file uses only the Python standard library so it can be uploaded and invoked
with the GPU host's Python. It is a process supervisor, not a security sandbox.
"""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid


TERMINAL = frozenset({"completed", "failed", "timed_out", "cancelled", "lost"})
MAX_CONTROL_BYTES = 1024 * 1024
MAX_TAIL_BYTES = 64 * 1024
MAX_TRANSFER_BYTES = 8 * 1024**3
ALLOWED_ENV = frozenset({"RECLAB_SEED", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED",
                         "PYTORCH_CUDA_ALLOC_CONF", "CUBLAS_WORKSPACE_CONFIG"})


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
        raise ValueError("job_id must contain 1-96 ASCII letters, digits, underscores or hyphens")
    return value


def _ordinary(info):
    return not (stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400)


def _root(value, *, create=False):
    if not isinstance(value, str) or not value or any(c in value for c in "\x00\r\n"):
        raise ValueError("root must be an absolute directory")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("root must be an absolute directory")
    if any(part in {".", ".."} for part in value.replace("\\", "/").split("/")):
        raise ValueError("root cannot contain traversal")
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if not _ordinary(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("root components must be ordinary directories")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("root directory does not exist")
    return path


def _path(root, relative, *, allow_dot=False):
    if relative == "." and allow_dot:
        return _root(str(root))
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("path must be a POSIX relative path")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if (posix.is_absolute() or windows.drive or windows.root
            or any(p in {"", ".", ".."} for p in relative.split("/"))):
        raise ValueError("path escapes root")
    _root(str(root))
    candidate = root.joinpath(*posix.parts)
    for i in range(1, len(posix.parts) + 1):
        part = root.joinpath(*posix.parts[:i])
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if not _ordinary(info):
            raise ValueError("symlinks and reparse points are forbidden")
        if i < len(posix.parts) and not stat.S_ISDIR(info.st_mode):
            raise ValueError("path component is not a directory")
    return candidate


def _read(path, limit=MAX_CONTROL_BYTES):
    info = path.lstat()
    if not _ordinary(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
        raise ValueError("expected a bounded, single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("file identity changed")
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file exceeds size limit")
    return data


def _json(path):
    return json.loads(_read(path))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _atomic(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def _lock(path):
    if os.name != "posix":
        raise RuntimeError("remote job supervision requires Linux")
    import fcntl
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _identity(pid, *, include_zombie=False):
    """Return Linux process start ticks, rejecting zombies and PID reuse."""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return None if fields[0] == "Z" and not include_zombie else fields[19]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def _alive(pid, started):
    return started is not None and _identity(pid) == str(started)


def _positive(value, name, maximum):
    if type(value) not in {int, float} or not 0 < value <= maximum:
        raise ValueError(f"{name} must be positive and at most {maximum}")
    return value


def _public_path(root, relative, *, write=False):
    path = _path(root, relative)
    if relative.split("/")[0] == ".control" or (write and relative.split("/")[0] == "jobs"):
        raise ValueError("path is reserved for the remote helper")
    return path


def _file_digest(path):
    info = path.lstat()
    if not _ordinary(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("expected a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("file identity changed")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
        raise ValueError("file changed while hashing")
    return {"size": info.st_size, "sha256": digest.hexdigest()}


def _job(root, job_id, *, create=False):
    job_id = _identifier(job_id)
    jobs = _path(root, "jobs")
    if create:
        jobs.mkdir(mode=0o700, exist_ok=True)
    directory = _path(root, "jobs/" + job_id)
    if create:
        directory.mkdir(mode=0o700, exist_ok=True)
    if not directory.is_dir():
        raise ValueError("unknown job_id")
    return directory


def _job_lock(directory):
    return _lock(_path(directory, ".lock"))


def _update(directory, **fields):
    with _job_lock(directory):
        path = _path(directory, "status.json")
        value = _json(path) if path.exists() else {}
        value.update(fields)
        value["updated_at"] = time.time()
        _atomic(path, value)
        return value


def _request(root, payload):
    job_id = _identifier(payload.get("job_id"))
    argv = payload.get("argv")
    if (not isinstance(argv, list) or not 2 <= len(argv) <= 256
            or any(not isinstance(a, str) or not a or "\x00" in a or len(a) > 32768 for a in argv)):
        raise ValueError("argv must contain a Python executable and script")
    if not Path(argv[0]).is_absolute() or os.path.realpath(argv[0]) != os.path.realpath(sys.executable):
        raise ValueError("argv must use this helper's explicit Python executable")
    position = 1
    while position < len(argv) and argv[position] in {"-B", "-u"}:
        position += 1
    if position == len(argv) or argv[position].startswith("-"):
        raise ValueError("only Python script execution is permitted; -c and -m are forbidden")
    cwd_relative = payload.get("cwd", ".")
    cwd = _path(root, cwd_relative, allow_dot=True)
    if not cwd.is_dir():
        raise ValueError("cwd must be a directory within root")
    script = Path(argv[position])
    if script.is_absolute():
        try:
            script_relative = script.relative_to(root).as_posix()
        except ValueError:
            raise ValueError("Python script must be within root") from None
    else:
        script_relative = (PurePosixPath(cwd_relative) / argv[position]).as_posix()
    script = _public_path(root, script_relative)
    _file_digest(script)
    requested_env = payload.get("env", {})
    if (not isinstance(requested_env, dict) or set(requested_env) - ALLOWED_ENV
            or any(not isinstance(v, str) or len(v) > 4096 or "\x00" in v for v in requested_env.values())):
        raise ValueError("env contains unsupported names or values")
    timeout = _positive(payload.get("timeout_seconds", 600), "timeout_seconds", 7 * 86400)
    normalized_argv = list(argv)
    normalized_argv[0] = sys.executable
    normalized_argv[position] = str(script)
    return {"job_id": job_id, "argv": normalized_argv, "cwd": str(cwd),
            "env": requested_env, "timeout_seconds": timeout}


def _session(root, job_id):
    return "reclab_" + hashlib.sha256((str(root) + "/" + job_id).encode()).hexdigest()[:32]


def _tmux_alive(session):
    try:
        return subprocess.run(["tmux", "has-session", "-t", "=" + session],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _status(root, job_id):
    directory = _job(root, job_id)
    with _job_lock(directory):
        path = _path(directory, "status.json")
        value = _json(path)
        if value["status"] not in TERMINAL:
            wrapper_alive = _alive(value.get("wrapper_pid"), value.get("wrapper_start"))
            if not wrapper_alive and not _tmux_alive(value["session"]):
                value.update(status="lost", exit_code=None, stop_confirmed=False, finished_at=time.time(),
                             updated_at=time.time(), error="supervisor disappeared without an exit record")
                _atomic(path, value)
        return value


def _run(root, payload):
    if os.name != "posix" or not Path("/proc/self/stat").exists():
        raise RuntimeError("remote execution requires Linux /proc and tmux")
    request = _request(root, payload)
    directory = _job(root, request["job_id"], create=True)
    fingerprint = hashlib.sha256(_canonical(request)).hexdigest()
    with _job_lock(directory):
        request_path = _path(directory, "request.json")
        if request_path.exists():
            if hashlib.sha256(_canonical(_json(request_path))).hexdigest() != fingerprint:
                raise ValueError("job_id already belongs to a different request")
            return _json(_path(directory, "status.json"))
        _atomic(request_path, request)
        value = {"job_id": request["job_id"], "status": "starting", "exit_code": None,
                 "stop_confirmed": False,
                 "request_hash": fingerprint, "session": _session(root, request["job_id"]),
                 "created_at": time.time(), "updated_at": time.time(),
                 "log_path": "jobs/" + request["job_id"] + "/job.log"}
        _atomic(_path(directory, "status.json"), value)
        try:
            # tmux accepts argv directly with multiple command arguments. No shell
            # command is interpolated from the submitted model argv.
            process = subprocess.run(["tmux", "new-session", "-d", "-s", value["session"],
                                      "-c", str(root), sys.executable, "-B", str(Path(__file__).absolute()),
                                      "_execute", str(root), request["job_id"]],
                                     capture_output=True, text=True, timeout=20)
            if process.returncode:
                raise RuntimeError("tmux launch failed: " + process.stderr[-2000:])
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            # A timed-out tmux client may have created the session: retain an
            # explicit uncertain state for status reconciliation, never retry it.
            if isinstance(error, subprocess.TimeoutExpired):
                value["error"] = "tmux launch response timed out; query this same job_id"
            else:
                value.update(status="failed", error=str(error), finished_at=time.time(),
                             stop_confirmed=not _tmux_alive(value["session"]))
            _atomic(_path(directory, "status.json"), value)
        return value


def _environment(requested):
    env = {"PATH": str(Path(sys.executable).parent) + ":/usr/local/bin:/usr/bin:/bin",
           "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
    for name in ("LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES",
                 "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"):
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(requested)
    return env


def _signal_group(value, sig):
    pid, started = value.get("pid"), value.get("process_start")
    if started is None or _identity(pid, include_zombie=True) != str(started):
        return False
    try:
        if os.getpgid(pid) != pid:
            raise RuntimeError("job process group identity does not match")
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def _group_active(pgid):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            if fields[0] != "Z" and int(fields[2]) == pgid and int(fields[3]) == pgid:
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def _exit_pending(process):
    # WNOWAIT keeps the group leader's /proc identity until the whole group has
    # been cleaned up, including children that outlive their parent.
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _terminate(process, identity, grace=5):
    value = {"pid": process.pid, "process_start": identity}
    _signal_group(value, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while _group_active(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_active(process.pid):
        _signal_group(value, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while _group_active(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if _group_active(process.pid):
        raise RuntimeError("job process group did not stop")
    return process.wait(timeout=10)


def _execute(root_value, job_id):
    root = _root(root_value)
    directory = _job(root, job_id)
    request = _json(_path(directory, "request.json"))
    process = None
    exit_code = None
    terminal, error = "failed", None
    cancel = _path(directory, "cancel.json")
    started = time.monotonic()
    try:
        # Claim supervision and check cancellation under one lock. Otherwise a
        # stop arriving between these steps could confirm an unstarted job dead
        # just before this wrapper launches its child.
        with _job_lock(directory):
            status_path = _path(directory, "status.json")
            value = _json(status_path)
            if cancel.exists():
                terminal = "cancelled"
                return
            value.update(wrapper_pid=os.getpid(), wrapper_start=_identity(os.getpid()), updated_at=time.time())
            _atomic(status_path, value)
        log_path = _path(directory, "job.log")
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb", buffering=0) as log:
            process = subprocess.Popen(request["argv"], cwd=request["cwd"], env=_environment(request["env"]),
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
            identity = _identity(process.pid, include_zombie=True)
            _update(directory, status="running", started_at=time.time(), pid=process.pid,
                    process_start=identity, pgid=process.pid)
            while not _exit_pending(process):
                if cancel.exists():
                    terminal = "cancelled"
                    exit_code = _terminate(process, identity)
                    break
                if time.monotonic() - started >= request["timeout_seconds"]:
                    terminal = "timed_out"
                    exit_code = _terminate(process, identity)
                    break
                time.sleep(0.1)
            else:
                exit_code = (_terminate(process, identity) if _group_active(process.pid)
                             else process.wait(timeout=10))
                terminal = "cancelled" if cancel.exists() else ("completed" if exit_code == 0 else "failed")
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        if process is not None and process.returncode is None:
            try:
                exit_code = _terminate(process, _identity(process.pid, include_zombie=True))
            except Exception as termination_error:
                error += "; termination failed: " + str(termination_error)
    finally:
        stopped = process is None or not _group_active(process.pid)
        _update(directory, status=terminal, exit_code=exit_code, error=error,
                stop_confirmed=stopped, finished_at=time.time())


def _stop(root, job_id):
    directory = _job(root, job_id)
    with _job_lock(directory):
        value = _json(_path(directory, "status.json"))
        if value["status"] in TERMINAL and value.get("stop_confirmed"):
            return value
        _atomic(_path(directory, "cancel.json"), {"requested_at": time.time()})
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        value = _status(root, job_id)
        if value["status"] in TERMINAL and value.get("stop_confirmed"):
            return value
        if not _alive(value.get("wrapper_pid"), value.get("wrapper_start")):
            break
        time.sleep(0.1)
    # If the supervisor is stuck or gone, the control command still stops only
    # the recorded process group after checking its Linux start-time identity.
    if _signal_group(value, signal.SIGTERM):
        time.sleep(0.2)
        _signal_group(value, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while value.get("pid") and _group_active(value["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    if value.get("pid") and _group_active(value["pid"]):
        raise RuntimeError("job group has not stopped")
    if _alive(value.get("wrapper_pid"), value.get("wrapper_start")):
        raise RuntimeError("supervisor has not acknowledged stop; query status again")
    return _update(directory, status="cancelled", exit_code=None, stop_confirmed=True, finished_at=time.time())


def _tail(root, payload):
    directory = _job(root, payload.get("job_id"))
    lines = payload.get("lines", 100)
    if type(lines) is not int or not 1 <= lines <= 1000:
        raise ValueError("lines must be an integer between 1 and 1000")
    path = _path(directory, "job.log")
    if not path.exists():
        return {"text": "", "truncated": False, "bytes": 0}
    info = path.lstat()
    if not _ordinary(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("log must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("log file identity changed")
        offset = max(0, opened.st_size - MAX_TAIL_BYTES)
        handle.seek(offset)
        data = handle.read(MAX_TAIL_BYTES)
    selected = b"\n".join(data.splitlines()[-lines:])
    return {"text": selected.decode("utf-8", errors="replace"), "bytes": len(selected),
            "truncated": offset > 0 or len(data.splitlines()) > lines}


def _command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    return {"exit_code": result.returncode, "stdout": result.stdout[-65536:], "stderr": result.stderr[-4096:]}


def _transfer_directory(root):
    path = _path(root, ".control/transfers")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _transfer_file(root, value):
    if not isinstance(value, str):
        raise ValueError("staging_path is required")
    directory = _transfer_directory(root)
    path = Path(value)
    if path.parent != directory or not re.fullmatch(r"[0-9a-f]{32}", path.name):
        raise ValueError("staging_path must name a helper-issued transfer")
    return _path(root, path.relative_to(root).as_posix())


def _transfer_description(payload):
    size, digest = payload.get("size"), payload.get("sha256")
    if type(size) is not int or not 0 <= size <= MAX_TRANSFER_BYTES:
        raise ValueError("size is outside the transfer limit")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("sha256 must be 64 lower-case hexadecimal digits")
    return {"size": size, "sha256": digest}


def _prepare_upload(root, payload):
    relative = payload.get("path")
    path = _public_path(root, relative, write=True)
    description = _transfer_description(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = _transfer_directory(root) / uuid.uuid4().hex
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    _atomic(stage.with_suffix(".json"), {"kind": "upload", "path": relative, **description})
    return {"staging_path": str(stage), **description}


def _commit_upload(root, payload):
    stage = _transfer_file(root, payload.get("staging_path"))
    relative = payload.get("path")
    path = _public_path(root, relative, write=True)
    description = _transfer_description(payload)
    expected = {"kind": "upload", "path": relative, **description}
    if _json(stage.with_suffix(".json")) != expected:
        raise ValueError("transfer does not match its prepared request")
    if stage.exists():
        if _file_digest(stage) != description:
            raise ValueError("uploaded file size or sha256 does not match")
        if path.exists():
            _file_digest(path)
        os.replace(stage, path)
    elif not path.exists() or _file_digest(path) != description:
        raise ValueError("uploaded staging file is missing")
    return {"path": str(path), **description}


def _prepare_download(root, payload):
    source = _public_path(root, payload.get("path"))
    maximum = payload.get("max_bytes")
    if type(maximum) is not int or not 0 <= maximum <= MAX_TRANSFER_BYTES:
        raise ValueError("max_bytes is required and must be within the transfer limit")
    info = source.lstat()
    if (not _ordinary(info) or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1 or info.st_size > maximum):
        raise ValueError("download must be a bounded single-link regular file")
    stage = _transfer_directory(root) / uuid.uuid4().hex
    read_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(read_fd, "rb") as reader:
            opened = os.fstat(reader.fileno())
            if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("download identity changed")
            descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as writer:
                digest, count = hashlib.sha256(), 0
                while True:
                    chunk = reader.read(min(1024 * 1024, maximum - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > maximum:
                        raise ValueError("download grew beyond max_bytes")
                    writer.write(chunk)
                    digest.update(chunk)
            after = os.fstat(reader.fileno())
            if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("download changed during snapshot")
        stage.chmod(0o400)
        description = {"size": count, "sha256": digest.hexdigest()}
        _atomic(stage.with_suffix(".json"), {"kind": "download", **description})
        return {"staging_path": str(stage), **description}
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def dispatch(payload):
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    operation = payload.get("op")
    root = _root(payload.get("root"), create=operation == "hello")
    if operation == "hello":
        versions = {}
        for package in ("torch", "numpy"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        torch_cuda = None
        if versions["torch"]:
            runtime = _command([sys.executable, "-c", "import torch; print(torch.version.cuda or '')"])
            if runtime["exit_code"]:
                raise RuntimeError("cannot inspect installed PyTorch runtime: " + runtime["stderr"])
            torch_cuda = runtime["stdout"].strip() or None
        return {"protocol_version": 1, "root": str(root), "hostname": socket.gethostname(), "python": sys.executable,
                "python_version": sys.version, "platform": sys.platform,
                "helper_sha256": _file_digest(Path(__file__))["sha256"],
                "libraries": versions, "torch_cuda": torch_cuda, "tmux": _command(["tmux", "-V"]),
                "gpu": _command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                  "--format=csv,noheader,nounits"])}
    if operation == "gpu_status":
        return _command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
                         "--format=csv,noheader,nounits"])
    if operation == "mkdir":
        path = _public_path(root, payload.get("path"), write=True)
        path.mkdir(parents=True, exist_ok=True)
        return {"path": str(path)}
    if operation == "stat":
        path = _public_path(root, payload.get("path"))
        result = {"path": str(path), **_file_digest(path)}
        if "max_bytes" in payload:
            maximum = payload["max_bytes"]
            if type(maximum) is not int or maximum < 0 or result["size"] > maximum:
                raise ValueError("file exceeds max_bytes")
        return result
    if operation == "prepare_upload":
        return _prepare_upload(root, payload)
    if operation == "commit_upload":
        return _commit_upload(root, payload)
    if operation == "prepare_download":
        return _prepare_download(root, payload)
    if operation == "release_transfer":
        stage = _transfer_file(root, payload.get("staging_path"))
        stage.unlink(missing_ok=True)
        stage.with_suffix(".json").unlink(missing_ok=True)
        return {"released": True}
    if operation == "run":
        return _run(root, payload)
    if operation == "status":
        return _status(root, payload.get("job_id"))
    if operation == "stop":
        return _stop(root, payload.get("job_id"))
    if operation == "tail":
        return _tail(root, payload)
    raise ValueError("unknown operation")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 3 and argv[0] == "_execute":
        _execute(argv[1], argv[2])
        return 0
    if argv != ["control"]:
        raise SystemExit("usage: remote_agent.py control (JSON on stdin)")
    try:
        raw = sys.stdin.buffer.read(MAX_CONTROL_BYTES + 1)
        if len(raw) > MAX_CONTROL_BYTES:
            raise ValueError("control request is too large")
        result = {"ok": True, "result": dispatch(json.loads(raw))}
    except Exception as error:
        code = ("INVALID_ARGUMENTS" if isinstance(error, (ValueError, TypeError)) else
                "NOT_FOUND" if isinstance(error, FileNotFoundError) else "REMOTE_ERROR")
        result = {"ok": False, "error": {"code": code, "message": str(error)}}
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
