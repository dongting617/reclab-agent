"""Explicit historical baseline import with source, runtime and artifact checks."""
from __future__ import annotations
import hashlib
from dataclasses import replace
from pathlib import Path
from reclab_agent.infrastructure.files import atomic_write, confined_path, read_regular_file
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.research import ExperimentRecord
from reclab_agent.experiments.outcomes import scored_record


def baseline_identity(config, inspection):
    policy = config.execution_policy_path or config.execution_connection_path
    return {"source": inspection["source"]["tree_sha256"],
            "data": {k: inspection["dataset"][k] for k in ("dataset_id", "data_hash", "public_tree_sha256")},
            "protocol": inspection["evaluation"]["protocol_hash"],
            "seed": config.seeds[0], "backend": config.execution_backend,
            "policy_hash": hashlib.sha256(read_regular_file(policy, max_bytes=1_000_000)).hexdigest() if policy else None,
            "max_job_seconds": config.task.budget.max_job_seconds}


def import_baseline(reference, config, plan, inspection, research, output):
    path = Path(reference).absolute()
    saved = read_json_strict(path, max_bytes=32_000_000)
    if saved.get("baseline_identity") != baseline_identity(config, inspection):
        raise ValueError("historical baseline data/source/seed/runtime/training limits differ")
    candidates = [x for x in saved["experiments"]
                  if x["record"]["candidate_id"] == saved["baseline_id"]]
    if len(candidates) != 1:
        raise ValueError("historical baseline must identify exactly one experiment")
    old = candidates[0]["record"]
    entry = scored_record(path.parent, old, plan)
    if "metrics" not in entry:
        raise ValueError("historical baseline has no verified independent score")
    snapshot = research.workspace.snapshot(research.snapshots_root)
    if old["source_hash"] != snapshot["source_hash"]:
        raise ValueError("historical baseline source differs")
    # A report alone is not enough: require an actual checkpoint and verify the
    # immutable manifest written by a prior successful export.
    manifest = saved.get("baseline_artifacts")
    if not manifest or not any(p.endswith((".pt", ".pth")) for p in manifest):
        raise ValueError("historical baseline checkpoint manifest missing")
    for relative, digest in manifest.items():
        raw = read_regular_file(confined_path(path.parent, relative), max_bytes=2_000_000_000)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("historical baseline artifact integrity mismatch")
        atomic_write(confined_path(output, relative), raw)
    grade = read_regular_file(confined_path(path.parent, old["score"]["source"]), max_bytes=512_000_000)
    atomic_write(confined_path(output, old["score"]["source"]), grade)
    record = replace(ExperimentRecord.from_dict(old), candidate_id=config.task.baseline_id)
    service = getattr(research, "service", research)
    service._records[record.experiment_id] = record
    return {**scored_record(output, record.to_dict(), plan), "reused_from": str(path)}


def checkpoint_manifest(output, entry):
    root = confined_path(output, "experiments/" + entry["record"]["experiment_id"])
    result = {}
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".pt", ".pth", ".json", ".jsonl"}:
                relative = path.relative_to(output).as_posix()
                raw = read_regular_file(confined_path(output, relative), max_bytes=2_000_000_000)
                result[relative] = hashlib.sha256(raw).hexdigest()
    return result
