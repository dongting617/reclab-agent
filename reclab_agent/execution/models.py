"""Execution contracts and administrator-approved input identities."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
from reclab_agent.domain.contracts import content_hash, identifier
from reclab_agent.infrastructure.files import (
    confined_path,
    read_regular_file as _regular,
    validated_directory as _directory,
)


class CapabilityError(RuntimeError):
    """No suitable local Docker execution capability; no host fallback exists."""


class DockerError(RuntimeError):
    pass


class OwnershipError(DockerError):
    """A name exists without this worker's exact job ownership labels."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class InputTree:
    directory: Path
    sha256: str
    kind: str

    @classmethod
    def freeze(cls, directory: str | Path, kind: str) -> "InputTree":
        if kind not in {"source", "public_data"}:
            raise ValueError("unsupported input kind")
        directory = _directory(directory)
        hashes: dict[str, str] = {}
        # No automatic exclusion: an unexpected file is an error, not hidden input.
        for parent, directories, files in os.walk(directory, followlinks=False):
            for name in directories:
                _directory(Path(parent) / name)
            for name in files:
                path = Path(parent) / name
                relative = path.relative_to(directory).as_posix()
                confined_path(directory, relative)
                if kind == "public_data":
                    if relative not in {"manifest.json", "train.json", "valid.json"}:
                        raise ValueError("public data contains an unregistered file; private/test files cannot be mounted")
                elif (not relative.startswith(("src/", "configs/"))
                      or path.suffix not in {".py", ".json", ".yaml", ".yml"}):
                    raise ValueError("source snapshot must contain only registered src/configs text")
                hashes[relative] = hashlib.sha256(_regular(path, max_bytes=1_000_000 if kind == "source" else 512_000_000)).hexdigest()
                if len(hashes) > 4096:
                    raise ValueError("too many input files")
        required = {"src/train.py", "src/predict.py"} if kind == "source" else {"manifest.json", "train.json", "valid.json"}
        if not required <= hashes.keys():
            raise ValueError("input snapshot is missing required files")
        return cls(directory, content_hash(hashes), kind)

    def verify(self) -> None:
        if InputTree.freeze(self.directory, self.kind) != self:
            raise ValueError("registered input changed after freezing")


@dataclass(frozen=True)
class ResourceLimits:
    cpus: float = 2.0
    memory_mb: int = 2048
    pids: int = 128
    seconds: int = 600

    def __post_init__(self):
        if isinstance(self.cpus, bool) or not isinstance(self.cpus, (int, float)) or not math.isfinite(self.cpus) or not 0.1 <= self.cpus <= 64:
            raise ValueError("cpus must be finite and between 0.1 and 64")
        for name, minimum, maximum in (("memory_mb", 64, 131072), ("pids", 8, 1024), ("seconds", 1, 86400)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} is outside the permitted range")


@dataclass(frozen=True)
class JobSpec:
    source_id: str
    dataset_id: str
    phase: str = "train"
    config: str = "default"
    seed: int = 42
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    model_job_id: str | None = None

    def __post_init__(self):
        for value in (self.source_id, self.dataset_id, self.config):
            identifier(value)
        if self.phase not in {"train", "predict"} or not isinstance(self.limits, ResourceLimits):
            raise ValueError("invalid phase or resource limits")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be a nonnegative 31-bit integer")
        if self.phase == "predict":
            identifier(self.model_job_id)
        elif self.model_job_id is not None:
            raise ValueError("training cannot mount another job's model")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JobSpec":
        payload = dict(value)
        payload["limits"] = ResourceLimits(**payload.get("limits", {}))
        return cls(**payload)


@dataclass(frozen=True)
class WorkerPolicy:
    image: str
    sources: Mapping[str, InputTree]
    datasets: Mapping[str, InputTree]
    max_limits: ResourceLimits = field(default_factory=lambda: ResourceLimits(4, 8192, 256, 1800))
    gpu: bool = False
    max_output_bytes: int = 512_000_000
    max_concurrent_jobs: int = 1

    def __post_init__(self):
        # The operator must pre-pull/build this exact digest. Mutable tags fail closed.
        if not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]*@sha256:[a-f0-9]{64}", self.image):
            raise ValueError("worker image must be an administrator-approved immutable repository digest")
        if type(self.gpu) is not bool or not isinstance(self.max_limits, ResourceLimits):
            raise ValueError("invalid worker capability policy")
        if type(self.max_output_bytes) is not int or not 1024 <= self.max_output_bytes <= 2_000_000_000:
            raise ValueError("invalid output collection bound")
        if type(self.max_concurrent_jobs) is not int or not 1 <= self.max_concurrent_jobs <= 4:
            raise ValueError("max_concurrent_jobs must be between 1 and 4")
        for name, kind in (("sources", "source"), ("datasets", "public_data")):
            registry = dict(getattr(self, name))
            if not registry:
                raise ValueError("explicit input registries are required")
            for key, value in registry.items():
                identifier(key)
                if not isinstance(value, InputTree) or value.kind != kind:
                    raise ValueError("wrong input tree kind")
                value.verify()
            object.__setattr__(self, name, MappingProxyType(registry))

    def identity(self) -> str:
        return content_hash({"image": self.image, "gpu": self.gpu, "limits": asdict(self.max_limits),
                             "max_output_bytes": self.max_output_bytes,
                             "max_concurrent_jobs": self.max_concurrent_jobs, "schema": 1})
