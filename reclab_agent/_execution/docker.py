"""Fixed local Docker CLI transport without a shell or candidate arguments."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

from ..files import atomic_write, validated_directory as _directory
from .models import CapabilityError, CommandResult, DockerError


class SubprocessDocker:
    """Only the fixed local Docker CLI is ever invoked; no shell or candidate argv."""

    def __init__(self, state_root: Path):
        self.executable = shutil.which("docker")
        self.config = _directory(state_root / "docker-client", create=True)
        atomic_write(self.config / "config.json", b'{"auths":{}}')

    def __call__(self, args: list[str], timeout: int = 10) -> CommandResult:
        if sys.platform != "linux" or not self.executable:
            raise CapabilityError("a dedicated Linux host with a local Docker daemon is required; host execution is disabled")
        environment = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                       "LANG": "C.UTF-8", "DOCKER_CONFIG": str(self.config)}
        try:
            completed = subprocess.run([self.executable, "--host", "unix:///var/run/docker.sock", *args],
                                       shell=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                       timeout=timeout, env=environment, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DockerError(f"Docker CLI unavailable or command outcome uncertain: {type(error).__name__}") from error
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
