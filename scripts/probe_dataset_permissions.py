"""Fixed, trusted Linux file-permission probe; NOT a candidate-code executor.

Creates a new read-only copy of exactly three public JSON files, then forks a
fixed probe which drops root privileges before attempting known file accesses.
Original files and permissions are never changed. No shell/code argument exists.
"""
from __future__ import annotations
import argparse
import codecs
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import stat
import sys
import tempfile
import time
from typing import Any


PUBLIC_FILES = frozenset({"manifest.json", "train.json", "valid.json"})
FORBIDDEN_KEYS = frozenset({"test", "test_target", "test_targets", "test_labels", "test_raw_item_id"})
UNPRIVILEGED_ID = 65534
MAX_REPORT_BYTES = 65536
LIMITS = [
    "Only fixed trusted probe code is executed; no generated/candidate code is permitted.",
    "Evidence covers this UID's access to the named files at probe time, not a full sandbox.",
    "No namespace/chroot, network, process, CPU, memory, disk or GPU isolation is established.",
    "No Worker timeout/cancellation/recovery guarantees are established.",
    "Docker/container acceptance for week two remains incomplete.",
]


class ProbeError(ValueError):
    """Fail-closed input, platform or permission validation failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_links(path: Path) -> Path:
    if not path.is_absolute():
        raise ProbeError("all input paths must be absolute")
    if ".." in path.parts:
        raise ProbeError("input paths must not contain parent traversal")
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ProbeError("input paths must not contain symlinks/reparse points")
    return path


def _reject_private_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ProbeError("public JSON contains a forbidden test-label key")
            if key.lower() == "test_labels_exposed" and child is not False:
                raise ProbeError("test_labels_exposed must be false when present")
            _reject_private_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_private_keys(child)


def validate_public(public_dir: Path) -> dict[str, str]:
    """Read-only, cross-platform validation; does not claim OS isolation."""
    _no_links(public_dir)
    if not public_dir.is_dir() or {p.name for p in public_dir.iterdir()} != PUBLIC_FILES:
        raise ProbeError("public directory must contain exactly manifest/train/valid JSON files")
    hashes: dict[str, str] = {}
    for name in sorted(PUBLIC_FILES):
        path = _no_links(public_dir / name)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ProbeError("public entries must be regular files")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ProbeError("public JSON files must contain objects")
        _reject_private_keys(value)
        hashes[name] = _sha256(path)
    return hashes


def _platform_preflight() -> None:
    if sys.platform != "linux" or not hasattr(os, "fork"):
        raise ProbeError("a real Linux root process with fork/prctl is required")
    if os.geteuid() != 0 or os.getuid() != 0:
        raise ProbeError("parent must be root so original private-file existence can be verified")


def _private_file(path: Path) -> dict[str, Any]:
    _no_links(path)
    if Path("/root") not in path.parents or not stat.S_ISREG(path.stat().st_mode):
        raise ProbeError("private/source paths must be existing regular files below /root")
    guard = Path("/root").stat()
    if guard.st_uid != 0 or stat.S_IMODE(guard.st_mode) != 0o700:
        raise ProbeError("/root must already be root-owned mode 0700; probe never changes it")
    return {"path": str(path), "verified_exists_by_root": True,
            "sha256": _sha256(path), "bytes": path.stat().st_size,
            "guard": {"path": "/root", "uid": guard.st_uid, "mode": "0700"}}


def _status() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(":")
        if key in {"Uid", "Gid", "Groups", "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs"}:
            values[key] = value.strip()
    return values


def _prctl(libc: Any, option: int, arg: int) -> None:
    if libc.prctl(option, arg, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _drop_privileges() -> dict[str, Any]:
    # All imports, source loading and stage preparation precede this fixed child.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    os.environ.clear()
    _prctl(libc, 38, 1)  # PR_SET_NO_NEW_PRIVS
    _prctl(libc, 47, 4)  # PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL
    for capability in range(64):
        try:
            _prctl(libc, 24, capability)  # PR_CAPBSET_DROP
        except OSError as exc:
            if exc.errno != errno.EINVAL:
                raise
    os.setgroups([])
    os.setresgid(UNPRIVILEGED_ID, UNPRIVILEGED_ID, UNPRIVILEGED_ID)
    os.setresuid(UNPRIVILEGED_ID, UNPRIVILEGED_ID, UNPRIVILEGED_ID)
    state = _status()
    if os.getresuid() != (UNPRIVILEGED_ID,) * 3 or os.getresgid() != (UNPRIVILEGED_ID,) * 3:
        raise ProbeError("real/effective/saved UID/GID were not fully dropped")
    if os.getgroups() or os.environ or state.get("NoNewPrivs") != "1":
        raise ProbeError("groups/environment/no_new_privs postcondition failed")
    for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if name not in state or int(state[name], 16) != 0:
            raise ProbeError("all capability sets must be zero")
    state["environment_entries"] = len(os.environ)
    state["nonstandard_fds"] = sorted(int(p.name) for p in Path("/proc/self/fd").iterdir()
                                      if p.name.isdigit() and int(p.name) > 2
                                      and _fd_open(int(p.name)))
    if state["nonstandard_fds"]:
        raise ProbeError("inherited nonstandard descriptors remain open")
    return state


def _fd_open(fd: int) -> bool:
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


def _attempt(path: Path, *, write: bool = False) -> dict[str, Any]:
    try:
        # No O_TRUNC: an unexpected write permission does not corrupt originals.
        fd = os.open(path, os.O_WRONLY if write else os.O_RDONLY)
        try:
            if not write:
                os.read(fd, 1)  # Do not report file contents, even on failure.
        finally:
            os.close(fd)
        return {"allowed": True, "errno": None}
    except OSError as exc:
        return {"allowed": False, "errno": exc.errno, "errno_name": errno.errorcode.get(exc.errno)}


def _fixed_checks(stage: Path, private: Path, original: Path) -> dict[str, Any]:
    state = _drop_privileges()
    public = stage / "public"
    reads = {name: _attempt(public / name) for name in sorted(PUBLIC_FILES)}
    writes = {name: _attempt(public / name, write=True) for name in sorted(PUBLIC_FILES)}
    denied = {"private_test": _attempt(private), "original_source": _attempt(original),
              "parent_traversal": _attempt(public / os.path.relpath(private, public)),
              "probe_symlink": _attempt(stage / "private-test-link")}
    # Parent has proved these exact targets exist. EACCES, rather than ENOENT,
    # demonstrates the named root-only boundary instead of missing-file success.
    passed = (all(r["allowed"] for r in reads.values())
              and all(not r["allowed"] and r["errno"] == errno.EACCES for r in writes.values())
              and all(not r["allowed"] and r["errno"] == errno.EACCES for r in denied.values()))
    return {"status": "passed" if passed else "failed", "identity": state,
            "public_read": reads, "public_write_open": writes, "private_access": denied}


def _child(stage: Path, private: Path, original: Path, read_fd: int, write_fd: int) -> None:
    os.close(read_fd)
    os.dup2(write_fd, 1)
    # stdin/stderr must not carry an inherited terminal, credential socket or file.
    null_fd = os.open(os.devnull, os.O_RDWR)
    os.dup2(null_fd, 0)
    os.dup2(null_fd, 2)
    os.chdir("/")
    # Enumerating after fork also closes descriptors above a lowered RLIMIT.
    for name in os.listdir("/proc/self/fd"):
        if name.isdigit() and int(name) > 2:
            try:
                os.close(int(name))
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise
    try:
        result = _fixed_checks(stage, private, original)
    except BaseException as exc:
        result = {"status": "failed", "error_type": type(exc).__name__,
                  "error": "fixed child probe failed closed", "errno": getattr(exc, "errno", None)}
    payload = json.dumps(result, sort_keys=True).encode("utf-8")
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(1, payload[offset:])
    finally:
        os._exit(0 if result["status"] == "passed" else 1)


def _run_child(stage: Path, private: Path, original: Path) -> dict[str, Any]:
    # Conda may live below /root: resolving a codec for the first time after
    # setresuid would otherwise try importing encodings from an unreadable path.
    codecs.lookup("ascii")
    codecs.lookup("utf-8")
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        _child(stage, private, original, read_fd, write_fd)
    os.close(write_fd)
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + 15
    timed_out = False
    try:
        while True:
            wait = deadline - time.monotonic()
            if wait <= 0 or not select.select([read_fd], [], [], wait)[0]:
                timed_out = True
                os.kill(pid, signal.SIGKILL)
                break
            chunk = os.read(read_fd, 8192)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_REPORT_BYTES:
                os.kill(pid, signal.SIGKILL)
                raise ProbeError("fixed child report exceeded size limit")
            chunks.append(chunk)
    finally:
        os.close(read_fd)
        _, wait_status = os.waitpid(pid, 0)
    if timed_out:
        raise ProbeError("fixed probe exceeded 15 seconds; this is not a Worker timeout acceptance test")
    try:
        result = json.loads(b"".join(chunks))
    except (ValueError, UnicodeError) as exc:
        raise ProbeError("fixed child returned no valid report") from exc
    if not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0 or result.get("status") != "passed":
        return {"status": "failed", "child": result}
    return result


def run_probe(public_dir: Path, private_test_path: Path, original_source_path: Path) -> dict[str, Any]:
    _platform_preflight()
    public_hashes = validate_public(public_dir)
    private_info = _private_file(private_test_path)
    source_info = _private_file(original_source_path)
    if private_test_path == original_source_path:
        raise ProbeError("private test and original source must be distinct files")
    temporary_root = _no_links(Path("/tmp"))
    tmp_info = temporary_root.stat()
    if tmp_info.st_uid != 0 or not stat.S_ISDIR(tmp_info.st_mode):
        raise ProbeError("/tmp must be an existing root-owned directory")
    stage = Path(tempfile.mkdtemp(prefix="reclab-permission-probe-", dir=temporary_root))
    public = stage / "public"
    public.mkdir(mode=0o700)
    for name in sorted(PUBLIC_FILES):
        destination = public / name
        with destination.open("xb") as stream:
            stream.write((public_dir / name).read_bytes())
        destination.chmod(0o444)
        if _sha256(destination) != public_hashes[name]:
            raise ProbeError("public file changed while preparing the probe copy")
    (stage / "private-test-link").symlink_to(private_test_path)
    public.chmod(0o555)
    stage.chmod(0o555)
    result = _run_child(stage, private_test_path, original_source_path)
    after = {name: _sha256(public / name) for name in sorted(PUBLIC_FILES)}
    unchanged = (after == public_hashes
                 and _sha256(private_test_path) == private_info["sha256"]
                 and _sha256(original_source_path) == source_info["sha256"])
    return {"schema_version": 1, "kind": "fixed_trusted_file_permission_probe",
            "status": "passed" if result.get("status") == "passed" and unchanged else "failed",
            "sandbox_established": False, "docker_verified": False, "limits": LIMITS,
            "probe_source_sha256": _sha256(Path(__file__).absolute()),
            "public_source": str(public_dir), "stage": str(stage),
            "stage_retained": True, "public_sha256": public_hashes,
            "public_unchanged": unchanged, "public_directory_mode": "0555", "public_file_mode": "0444",
            "private_test": private_info, "original_source": source_info, "probe": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", required=True, type=Path)
    parser.add_argument("--private-test-path", required=True, type=Path)
    parser.add_argument("--original-source-path", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New report file; never overwrites an existing file")
    args = parser.parse_args(argv)
    if args.output is not None and (not args.output.is_absolute() or args.output.exists()):
        parser.error("output must be an absolute new file path")
    try:
        result = run_probe(args.public_dir, args.private_test_path, args.original_source_path)
    except (OSError, ValueError) as exc:
        result = {"schema_version": 1, "kind": "fixed_trusted_file_permission_probe",
                  "status": "failed", "sandbox_established": False, "docker_verified": False,
                  "limits": LIMITS, "error_type": type(exc).__name__, "error": str(exc)}
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
