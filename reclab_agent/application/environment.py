"""Host-only environment loading and the optional WSL/Windows SSH bridge."""
from __future__ import annotations
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import subprocess
import sys
from reclab_agent.infrastructure.files import read_regular_file
from reclab_agent.tools.remote import RemoteToolbox
from reclab_agent.execution.ssh import SSHConnection
from reclab_agent.domain.configuration import read_json_strict

SSH_ENV_FIELDS = {
    "SSH_HOST": "host", "SSH_PORT": "port", "SSH_USER": "user",
    "SSH_KEY_PATH": "key_path", "SSH_KNOWN_HOSTS_PATH": "known_hosts_path",
    "SSH_REMOTE_ROOT": "remote_root", "SSH_PYTHON": "python_executable",
}


def ssh_environment_values():
    """Read a complete connection without exposing values in errors or traces."""
    required = set(SSH_ENV_FIELDS) - {"SSH_KNOWN_HOSTS_PATH"}
    if any(not os.environ.get(name) for name in required):
        raise ValueError("configure SSH_HOST, SSH_PORT, SSH_USER, SSH_KEY_PATH, SSH_REMOTE_ROOT and SSH_PYTHON")
    value = {field: os.environ[name] for name, field in SSH_ENV_FIELDS.items() if os.environ.get(name)}
    if not re.fullmatch(r"[0-9]{1,5}", value["port"]):
        raise ValueError("SSH_PORT must be an integer between 1 and 65535")
    value["port"] = int(value["port"])
    for name in ("key_path", "known_hosts_path"):
        if name in value:
            text = value[name]
            if not (Path(text).is_absolute() or PureWindowsPath(text).is_absolute()):
                raise ValueError("SSH identity and known-hosts paths must be absolute")
    return value


def load_agent_environment(path, configuration) -> dict[str, bool]:
    """Load literal assignments atomically; never execute or expand shell text."""
    allowed = {configuration.model_env, configuration.api_key_env, "OPENAI_BASE_URL", *SSH_ENV_FIELDS}
    if hasattr(configuration, "tavily_key_env"):
        allowed.add(configuration.tavily_key_env)
    values = {}
    text = read_regular_file(Path(path), max_bytes=65_536).decode("utf-8-sig")
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*(.*)", line)
        if match is None:
            raise ValueError(f"invalid environment assignment on line {number}")
        name, value = match.groups()
        if name not in allowed:
            raise ValueError(f"unsupported environment variable on line {number}")
        if name in values:
            raise ValueError(f"duplicate environment variable on line {number}")
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0] or value[0] in value[1:-1]:
                raise ValueError(f"invalid quoted environment value on line {number}")
            value = value[1:-1]
        elif any(char.isspace() or char in "'\"" for char in value):
            raise ValueError(f"invalid environment value on line {number}")
        if "\x00" in value:
            raise ValueError(f"invalid environment value on line {number}")
        values[name] = value
    os.environ.update(values)
    return {name: bool(os.environ.get(name)) for name in sorted(allowed)
            if name not in SSH_ENV_FIELDS or name in os.environ}


def _is_wsl() -> bool:
    return sys.platform == "linux" and "microsoft" in platform.release().lower()


def _mounted_windows_path(value: str) -> str:
    """Translate a mounted drive path for Windows OpenSSH, preserving spaces."""
    match = re.fullmatch(r"/mnt/([A-Za-z])(?:/(.*))?", value)
    if match is None:
        raise ValueError("Windows SSH local paths must be under /mnt/<drive>")
    rest = match.group(2) or ""
    if any(part in {".", ".."} for part in rest.split("/")):
        raise ValueError("Windows SSH local paths cannot contain traversal")
    return match.group(1).upper() + ":\\" + rest.replace("/", "\\")


def _local_bridge_path(value, parent: Path) -> Path:
    if not isinstance(value, str) or not value or any(char in value for char in "\x00\r\n"):
        raise ValueError("SSH identity paths must be nonempty literal paths")
    windows = PureWindowsPath(value)
    if windows.drive:
        if not windows.is_absolute() or not re.fullmatch(r"[A-Za-z]:", windows.drive):
            raise ValueError("Windows SSH paths must use an absolute drive path")
        path = Path("/mnt") / windows.drive[0].lower() / Path(*windows.parts[1:])
    else:
        if "\\" in value:
            raise ValueError("invalid SSH identity path")
        path = Path(value)
        if not path.is_absolute():
            path = parent / path
    path = Path(os.path.abspath(path))
    _mounted_windows_path(str(path))
    return path


def _windows_command_runner(argv, **kwargs):
    argv = list(argv)
    if not argv or argv[0] not in {"ssh", "scp"}:
        raise ValueError("the Windows SSH bridge accepts ssh/scp only")
    argv[0] = "/mnt/c/Windows/System32/OpenSSH/" + argv[0] + ".exe"
    for index, value in enumerate(argv[1:], 1):
        if value.startswith("UserKnownHostsFile="):
            argv[index] = "UserKnownHostsFile=" + _mounted_windows_path(value.split("=", 1)[1])
        elif value.startswith("/mnt/"):
            argv[index] = _mounted_windows_path(value)
    return subprocess.run(argv, **kwargs)


def build_remote_tools(config, *, windows_ssh=False) -> RemoteToolbox | None:
    """Bind trusted transfer roots without connecting to SSH or reading keys."""
    if config.execution_backend == "docker":
        return None
    if config.execution_backend != "ssh_tmux":
        raise ValueError("configure an explicit ssh_tmux connection")
    runner = subprocess.run
    if windows_ssh:
        if not _is_wsl():
            raise ValueError("--windows-ssh requires WSL Linux")
        for executable in ("ssh.exe", "scp.exe"):
            if not Path("/mnt/c/Windows/System32/OpenSSH", executable).is_file():
                raise ValueError("Windows OpenSSH ssh.exe and scp.exe must be installed")
        configuration_path = config.execution_connection_path
        value = (read_json_strict(configuration_path, max_bytes=65_536)
                 if configuration_path is not None else ssh_environment_values())
        required = {"host", "port", "user", "key_path", "remote_root", "python_executable"}
        if (not isinstance(value, dict) or not required <= value.keys()
                or not value.keys() <= required | {"known_hosts_path"}):
            raise ValueError("SSH configuration must contain the documented fields only")
        for name in ("key_path", "known_hosts_path"):
            if name in value and value[name] is not None:
                value[name] = _local_bridge_path(value[name],
                    Path(configuration_path).absolute().parent if configuration_path is not None else Path.cwd())
        connection = SSHConnection(**value)
        runner = _windows_command_runner
    else:
        if config.execution_connection_path is not None:
            connection = SSHConnection.from_json(config.execution_connection_path)
        else:
            value = ssh_environment_values()
            for name in ("key_path", "known_hosts_path"):
                if name in value and os.name != "nt" and (PureWindowsPath(value[name]).drive or "\\" in value[name]):
                    raise ValueError("Windows SSH paths require --windows-ssh under WSL")
            connection = SSHConnection(**value)
    return RemoteToolbox(connection, upload_roots={
        "source": config.output_directory / "snapshots",
        "public": config.dataset_directory / "public",
    }, download_root=config.output_directory, command_runner=runner,
        max_timeout_seconds=config.task.budget.max_job_seconds)
