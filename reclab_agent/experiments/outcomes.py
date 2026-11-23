"""Verified experiment reads, comparisons and immutable source restoration."""
from __future__ import annotations
import hashlib
import json
import math
from reclab_agent.domain.contracts import content_hash
from reclab_agent.infrastructure.files import atomic_write, confined_path, read_regular_file
from reclab_agent.domain.research import ExperimentRecord


def scored_record(output, value, plan):
    record = ExperimentRecord.from_dict(value)
    result = {"record": record.to_dict()}
    if record.status != "completed":
        return result
    raw = read_regular_file(confined_path(output, record.score.source), max_bytes=512_000_000)
    if hashlib.sha256(raw).hexdigest() != record.score.sha256:
        raise ValueError("independent scoring artifact hash mismatch")
    grade = json.loads(raw)
    if grade["protocol_hash"] != plan.protocol_hash or record.seed != plan.seeds[0]:
        raise ValueError("independent score protocol or comparison seed mismatch")
    metrics = {name: grade["metrics"][name] for name in (plan.primary_metric, *plan.guardrail_metrics)}
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
        raise ValueError("independent scores must be finite")
    result["metrics"] = metrics
    result["evaluation_hash"] = grade.get("evaluation_hash")
    return result


def compare_experiments(baseline, candidate, primary, guards):
    before, after = baseline["record"], candidate["record"]
    for name in ("dataset_id", "protocol_hash", "seed"):
        if before[name] != after[name]:
            raise ValueError("incompatible experiment " + name)
    if baseline.get("evaluation_hash") != candidate.get("evaluation_hash"):
        raise ValueError("incompatible evaluation contexts")
    if before.get("image") != after.get("image") or before.get("runtime") != after.get("runtime"):
        raise ValueError("incompatible execution environments")
    deltas = {name: value - baseline["metrics"][name] for name, value in candidate["metrics"].items()}
    passes = all(deltas[name] >= 0 for name in guards)
    return {"candidate_id": after["candidate_id"], "experiment_id": after["experiment_id"],
            "deltas": deltas, "passes_guardrails": passes, "improves": deltas[primary] > 0 and passes}


def restore_snapshot(research, source_hash):
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("invalid source hash")
    root = confined_path(research.snapshots_root, source_hash)
    available = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    original = research.workspace.source_paths
    if any(not p.startswith("src/diagnostics/") for p in original.symmetric_difference(available)):
        raise ValueError("snapshot changed the registered source tree")
    # Match Workspace's universal-newline reads when verifying text identities.
    files = {p: read_regular_file(confined_path(root, p), max_bytes=1_000_000)
             .decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
             for p in sorted(available)}
    if content_hash(files) != source_hash:
        raise ValueError("snapshot source hash does not match its contents")
    # Read-only inputs must not change when restoring a candidate.
    for p in (research.workspace.source_paths - research.workspace.editable_paths) & available:
        if p.startswith("src/diagnostics/"):
            continue
        if files[p] != research.workspace.read(p)["text"]:
            raise ValueError("read-only source differs from snapshot")
    for p in original - available:
        confined_path(research.workspace.root, p).unlink()
    research.workspace.source_paths = frozenset(available)
    for p in available:
        atomic_write(confined_path(research.workspace.root, p), files[p].encode("utf-8"))
    return research.workspace.snapshot(research.snapshots_root)
