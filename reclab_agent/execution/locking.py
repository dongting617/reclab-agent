"""Shared process-safe, same-thread reentrant execution locks."""
from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import threading
from reclab_agent.domain.contracts import identifier
from reclab_agent.infrastructure.files import confined_path, validated_directory as _directory


_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[str, threading.RLock] = {}
_HELD_JOB_LOCKS = threading.local()


@contextmanager
def job_lock(root: Path, job_id: str):
    """Nonblocking lock whose stable file survives individual Worker processes."""
    identifier(job_id)
    directory = _directory(confined_path(root, "locks"), create=True)
    path = confined_path(directory, f"{job_id}.lock")
    key = os.path.normcase(str(path))
    with _LOCK_REGISTRY_GUARD:
        local_lock = _LOCK_REGISTRY.setdefault(key, threading.RLock())
    if not local_lock.acquire(blocking=False):
        yield False
        return
    held = getattr(_HELD_JOB_LOCKS, "paths", None)
    if held is None:
        held = _HELD_JOB_LOCKS.paths = set()
    descriptor = None
    locked = False
    try:
        if key in held:
            yield True
            return
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("job lock must be a private single-link regular file")
        if info.st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            yield False
            return
        locked = True
        held.add(key)
        yield True
    finally:
        if locked:
            held.remove(key)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        if descriptor is not None:
            os.close(descriptor)
        local_lock.release()
