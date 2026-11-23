"""Shared path confinement, atomic writes and bounded ordinary-file reads."""
from __future__ import annotations
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import uuid


def confined_path(root: str | Path, relative: str) -> Path:
    """Reject cross-platform traversal and all existing reparse/symlink components."""
    root = Path(os.path.abspath(os.fspath(root)))
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("path must be a non-empty POSIX relative path")
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if posix.is_absolute() or windows.drive or windows.root or any(p in {".", ".."} for p in relative.split("/")):
        raise ValueError("path escapes root")
    candidate = root.joinpath(*posix.parts)
    for path in (root, *[root.joinpath(*posix.parts[:i]) for i in range(1, len(posix.parts) + 1)]):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("symlinks and reparse points are not permitted")
    return candidate


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validated_directory(path: str | Path, *, create: bool = False) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if any(c in str(path) for c in (",", "\n", "\r", "\x00")):
        raise ValueError("Docker mount paths cannot contain commas or control characters")
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("symlink/reparse directory components are forbidden")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("expected a directory")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError("directory does not exist")
    return path


def read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    """Read only a bounded, single-link regular file (including on Windows)."""
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or getattr(info, "st_file_attributes", 0) & 0x400 or info.st_size > max_bytes):
        raise ValueError("artifact must be a bounded, single-link regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError("artifact changed during collection")
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("artifact exceeds size limit")
    return raw
