"""Audited, fixture-only Docker demonstration; never executes candidate code on host."""
from __future__ import annotations
from contextlib import closing, contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
from typing import Any
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.data.datasets import DatasetBundle, create_synthetic_bundle
from reclab_agent.execution.worker import (
    DockerError,
    DockerWorker,
    InputTree,
    JobSpec,
    ResourceLimits,
    WorkerPolicy,
)
from reclab_agent.memory.storage import atomic_write, confined_path
from reclab_agent.experiments.workspace import create_workspace


SEED = 42
LIMITS = ResourceLimits(cpus=1, memory_mb=256, pids=32, seconds=30)
RUN_SECONDS = 120
STOP_SECONDS = 30
SOURCE_ID = "audited-popularity-v1"

# These are text fixtures, never imported, eval'ed, or invoked by the controller.
TRAIN_SOURCE = '''"""Fixture-only popularity training, executed inside Docker."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()
seed = int(os.environ["RECLAB_SEED"])
random.seed(seed)
config = json.loads(Path(args.config).read_text())
assert config == {"algorithm": "popularity", "top_k": 20}
histories = json.loads((Path(args.data) / "train.json").read_text())
counts = Counter(item for history in histories.values() for item in history)
output = Path(args.output)
(output / "counts.json").write_text(json.dumps(dict(counts), sort_keys=True))
(output / "training.json").write_text(json.dumps({"seed": seed, "quality_evidence": "fixture_only"}))
'''
PREDICT_SOURCE = '''"""Fixture-only prediction, executed inside Docker; no host fallback."""
import argparse
import json
import os
from pathlib import Path
import random

parser = argparse.ArgumentParser()
for name in ("data", "output", "config", "model", "split"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
assert args.split == "valid"
seed = int(os.environ["RECLAB_SEED"])
random.seed(seed)
config = json.loads(Path(args.config).read_text())
assert config == {"algorithm": "popularity", "top_k": 20}
counts = json.loads((Path(args.model) / "counts.json").read_text())
manifest = json.loads((Path(args.data) / "manifest.json").read_text())
rows = json.loads((Path(args.data) / "valid.json").read_text())
catalog = sorted(manifest["catalog"], key=lambda item: (-counts.get(str(item), 0), item))
predictions = {user: [item for item in catalog if item not in set(row["history"])][:20]
               for user, row in rows.items()}
(Path(args.output) / "predictions.json").write_text(json.dumps(predictions, sort_keys=True))
'''
FILES = {"src/train.py": TRAIN_SOURCE, "src/predict.py": PREDICT_SOURCE,
         "configs/default.json": '{"algorithm":"popularity","top_k":20}\n'}


def _root(value: str | Path, *, exists: bool = True) -> Path:
    root = Path(os.path.abspath(os.fspath(value)))
    for path in reversed((root, *root.parents)):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("demo root cannot use symlink/reparse components")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("demo root must be a directory")
    if exists and not root.is_dir():
        raise ValueError("prepare a new demo root first")
    return root


def _write(root: Path, name: str, value: Any) -> None:
    atomic_write(confined_path(root, name), canonical_json(value).encode("utf-8"))


def prepare(root: str | Path) -> dict[str, Any]:
    root = _root(root, exists=False)
    # Reject even an empty existing directory: preparation never overwrites work.
    root.mkdir(parents=True, exist_ok=False)
    if os.name != "nt":
        root.chmod(0o700)
    workspace = create_workspace(root / "workspaces", SOURCE_ID, FILES)
    validation = workspace.validate()
    if not validation["ok"]:
        raise ValueError("audited demo template failed syntax validation")
    snapshot = workspace.snapshot(root / "snapshots")
    source = InputTree.freeze(snapshot["directory"], "source")
    bundle = create_synthetic_bundle(root / "dataset")
    public = InputTree.freeze(bundle.public, "public_data")
    result = {"schema_version": 1, "demo": SOURCE_ID, "quality_evidence": "fixture_only",
              "llm_connected": False, "seed": SEED, "max_jobs": 2, "max_concurrent_jobs": 1,
              "limits": asdict(LIMITS), "run_timeout_seconds": RUN_SECONDS,
              "stop_confirmation_timeout_seconds": STOP_SECONDS,
              "source_hash": snapshot["source_hash"], "source_tree_sha256": source.sha256,
              "snapshot": Path(snapshot["directory"]).relative_to(root).as_posix(),
              "dataset_id": bundle.dataset_id, "data_hash": bundle.data_hash,
              "public_tree_sha256": public.sha256,
              "mounted_data": "dataset/public", "test_labels_mounted": False,
              "validation": validation}
    _write(root, "demo.json", result)
    return result


def load(root: str | Path) -> tuple[Path, dict[str, Any], DatasetBundle, InputTree, InputTree]:
    root = _root(root)
    manifest = json.loads(confined_path(root, "demo.json").read_text(encoding="utf-8"))
    expected_hash = content_hash(dict(sorted(FILES.items())))
    if (manifest.get("schema_version") != 1 or manifest.get("demo") != SOURCE_ID
            or manifest.get("source_hash") != expected_hash
            or manifest.get("snapshot") != f"snapshots/{expected_hash}"
            or manifest.get("mounted_data") != "dataset/public"
            or manifest.get("limits") != asdict(LIMITS) or manifest.get("seed") != SEED):
        raise ValueError("demo manifest does not match the fixed audited fixture")
    snapshot = confined_path(root, manifest["snapshot"])
    for relative, text in FILES.items():
        if confined_path(snapshot, relative).read_text(encoding="utf-8") != text:
            raise ValueError("demo runs only the unchanged audited source snapshot")
    source = InputTree.freeze(snapshot, "source")
    bundle = DatasetBundle(confined_path(root, "dataset"))
    public = InputTree.freeze(bundle.public, "public_data")
    if (source.sha256 != manifest["source_tree_sha256"] or public.sha256 != manifest["public_tree_sha256"]
            or bundle.data_hash != manifest["data_hash"]
            or bundle.dataset_id != manifest["dataset_id"]
            or bundle.manifest.get("quality_evidence") != "fixture_only"):
        raise ValueError("frozen demo source or public data changed")
    return root, manifest, bundle, source, public


def policy_for(root: str | Path, image: str) -> tuple[WorkerPolicy, DatasetBundle, dict[str, Any]]:
    root, manifest, bundle, source, public = load(root)
    policy = WorkerPolicy(image, {SOURCE_ID: source}, {bundle.dataset_id: public},
                          max_limits=LIMITS, max_output_bytes=1_000_000, max_concurrent_jobs=1)
    existing = confined_path(root, "worker-policy.json")
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("image") != image:
        raise ValueError("a demo root is bound to one image; prepare another root to change it")
    return policy, bundle, manifest


def worker_for(root: str | Path, image: str) -> tuple[DockerWorker, DatasetBundle, dict[str, Any]]:
    policy, bundle, manifest = policy_for(root, image)
    root = _root(root)
    # Operator-readable policy for an independently supervised watchdog. This is
    # administrator configuration outside both candidate mounts.
    serialized = {"image": image, "gpu": False, "max_limits": asdict(LIMITS),
                  "max_output_bytes": policy.max_output_bytes, "max_concurrent_jobs": 1,
                  "sources": {key: {"directory": str(tree.directory), "sha256": tree.sha256}
                              for key, tree in policy.sources.items()},
                  "datasets": {key: {"directory": str(tree.directory), "sha256": tree.sha256}
                               for key, tree in policy.datasets.items()}}
    path = confined_path(root, "worker-policy.json")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != serialized:
        raise ValueError("existing demo worker policy changed; prepare another root")
    _write(root, "worker-policy.json", serialized)
    return DockerWorker(root / "worker", policy), bundle, manifest


@contextmanager
def run_lock(root: str | Path):
    """One demo controller per root; OS lock releases after a crash."""
    root = _root(root)
    path = confined_path(root, "controller.lock")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    acquired = False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("controller lock must be a regular single-link file")
        if info.st_size == 0:
            os.write(descriptor, b"0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _summary(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job.get(key) for key in ("job_id", "status", "exit_code", "reason", "receipt")}


def cached_jobs(root: str | Path) -> list[dict[str, Any]]:
    """Read DB records without polling/launching; never asserts container liveness."""
    database = confined_path(_root(root), "worker/jobs.sqlite3")
    if not database.exists():
        return []
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute(
            "SELECT job_id,status,exit_code,reason,receipt FROM docker_jobs ORDER BY created")]
    if len(rows) > 2:
        raise ValueError("demo state exceeded its two-job admission contract")
    for row in rows:
        row["receipt"] = json.loads(row["receipt"]) if row["receipt"] else None
        row["live_verified"] = False
    return rows


def refresh_failed_run(root: str | Path, error: dict[str, Any], stop: dict[str, Any]) -> None:
    """Reconcile the displayed job records after bounded cancellation, without launching."""
    root = _root(root)
    path = confined_path(root, "run.json")
    if not path.exists():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    observations = {row["job_id"]: row for row in stop.get("jobs", [])}
    jobs = cached_jobs(root)
    for job in jobs:
        if job["job_id"] in observations:
            job["stop_observation"] = observations[job["job_id"]]
    state.update(status="cancelled" if error.get("error") == "KeyboardInterrupt" else "failed",
                 error=error, jobs=jobs, jobs_state_source="worker_database_snapshot",
                 stop_confirmed=stop.get("stop_confirmed", False), stop=stop)
    _write(root, "run.json", state)


def execute(root: str | Path, image: str, *, worker: DockerWorker | None = None,
            monotonic=time.monotonic, sleep=time.sleep) -> dict[str, Any]:
    """Trusted child orchestration. The public CLI also enforces a process timeout."""
    root, manifest, bundle, _, _ = load(root)
    run_path = confined_path(root, "run.json")
    if run_path.exists():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous.get("image") != image:
            raise ValueError("a demo root is bound to one image; prepare another root to change it")
    state = {"schema_version": 1, "status": "running", "image": image, "seed": SEED,
             "quality_evidence": "fixture_only", "llm_connected": False,
             "source_hash": manifest["source_hash"], "source_tree_sha256": manifest["source_tree_sha256"],
             "data_hash": manifest["data_hash"], "public_tree_sha256": manifest["public_tree_sha256"],
             "jobs": [], "max_jobs": 2, "test_labels_mounted": False}
    _write(root, "run.json", state)
    if worker is None:
        worker, _, _ = worker_for(root, image)
    deadline = monotonic() + RUN_SECONDS
    worker.capabilities()  # Checks Linux, daemon, limits, seccomp, and pre-pulled digest.
    model_job_id = None
    for phase in ("train", "predict"):
        if monotonic() >= deadline:
            raise TimeoutError("demo total execution deadline exceeded")
        spec = JobSpec(SOURCE_ID, bundle.dataset_id, phase=phase, seed=SEED,
                       limits=LIMITS, model_job_id=model_job_id)
        job = worker.submit(spec)
        state["jobs"].append(_summary(job))
        _write(root, "run.json", state)
        while job["status"] not in DockerWorker.TERMINAL:
            if monotonic() >= deadline:
                raise TimeoutError("demo total execution deadline exceeded")
            sleep(0.2)
            job = worker.poll(job["job_id"])
            state["jobs"][-1] = _summary(job)
            _write(root, "run.json", state)
        if job["status"] != "completed":
            raise DockerError(f"{phase} job did not complete: {job['status']}")
        model_job_id = job["job_id"]
    predictions = worker.collect_predictions(model_job_id)
    report = bundle.grade(predictions, "valid")
    _write(root, "predictions.json", predictions)
    _write(root, "report.json", report)
    state.update(status="completed", predictions_sha256=content_hash(predictions),
                 report_sha256=content_hash(report), metrics=report["metrics"],
                 interpretation="Fixture contract demonstration only; no real model optimization or LLM agent success claim.")
    _write(root, "run.json", state)
    return state


def cancel_and_confirm(root: str | Path, image: str) -> dict[str, Any]:
    """Point-in-time stop observation, bounded by the public CLI's 30 s subprocess."""
    root = _root(root)
    ids = [job["job_id"] for job in cached_jobs(root)]
    results, request_errors = [], {}
    if ids:
        worker, _, _ = worker_for(root, image)
        # Every job needs a durable stop fence, including terminal DB records:
        # an administrator or delayed Docker command may have restarted it.
        for job_id in ids:
            try:
                worker._request_stop(job_id, "cancelled")
            except sqlite3.Error as error:
                # Worker writes its independent journal first. A busy database
                # must not prevent the watchdog inspecting/signalling the job.
                request_errors[job_id] = f"SQLite stop accounting pending: {error}"
        from reclab_agent.execution.watchdog import DockerDeadlineWatchdog
        watchdog = DockerDeadlineWatchdog(root / "worker", worker.policy, runner=worker.runner)
        stops = {row["job_id"]: row for row in watchdog.tick()}
        for job_id in ids:
            row = stops.get(job_id, {"job_id": job_id, "stop_confirmed": False,
                                     "detail": "no fresh container stop observation was returned"})
            row["observed_at_unix"] = time.time()
            row["confirmation_scope"] = "observation_only"
            if job_id in request_errors:
                row["stop_accounting_error"] = request_errors[job_id]
            results.append(row)
    result = {"stop_confirmed": all(row["stop_confirmed"] for row in results), "jobs": results,
              "observed_at_unix": time.time(), "confirmation_scope": "observation_only",
              "continued_watchdog_required": bool(ids),
              "detail": ("Only this observation confirms exit; continue the independent watchdog to catch late starts."
                         if ids else "no jobs were admitted")}
    _write(root, "cancel-observation.json", result)
    path = confined_path(root, "run.json")
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state.update(status="cancelled" if result["stop_confirmed"] else "stop_unconfirmed",
                     jobs=cached_jobs(root), jobs_state_source="worker_database_snapshot", stop=result)
        _write(root, "run.json", state)
    return result


def status(root: str | Path) -> dict[str, Any]:
    root, manifest, _, _, _ = load(root)
    path = confined_path(root, "run.json")
    return {"prepared": True, "manifest": manifest,
            "state_source": "cached_files_and_worker_database", "live_verified": False,
            "container_liveness": "unverified", "worker_jobs": cached_jobs(root),
            "run": json.loads(path.read_text(encoding="utf-8")) if path.exists() else None}
