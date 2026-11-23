"""Trusted administrator policy loading shared by execution entrypoints."""
from __future__ import annotations
import json
from pathlib import Path
from reclab_agent.execution.models import InputTree, ResourceLimits, WorkerPolicy


def load_policy(path: str | Path) -> WorkerPolicy:
    """Trusted configuration; paths/digests are never supplied through Agent tools."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"image", "sources", "datasets", "max_limits", "gpu", "max_output_bytes", "max_concurrent_jobs"}
    if not isinstance(value, dict) or not value.keys() <= allowed:
        raise ValueError("unknown policy field")
    for registry, kind in (("sources", "source"), ("datasets", "public_data")):
        value[registry] = {key: InputTree(Path(entry["directory"]), entry["sha256"], kind)
                           for key, entry in value[registry].items()}
    if "max_limits" in value:
        value["max_limits"] = ResourceLimits(**value["max_limits"])
    return WorkerPolicy(**value)
