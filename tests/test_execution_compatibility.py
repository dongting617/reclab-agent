"""Compatibility with persisted Worker identities from before module extraction."""
from dataclasses import asdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from reclab_agent import execution, files, storage, worker
from reclab_agent._execution import config, docker, models
from reclab_agent.contracts import canonical_json
from tests.support.docker import FakeDocker, IMAGE


LEGACY_SPEC_JSON = (
    '{"config":"default","dataset_id":"tiny","limits":{"cpus":2.0,'
    '"memory_mb":2048,"pids":128,"seconds":600},"model_job_id":null,'
    '"phase":"train","seed":42,"source_id":"baseline"}'
)
LEGACY_POLICY_HASH = "e504bd5bf9323834ccb8393ec62900e267049f5744e2b6fa2fe859d81ade3fe1"
LEGACY_JOB_KEY = "07b7991def2e396f507936e2a635b5de78a3f9150f264f4c287e567f1b281a83"
LEGACY_JOB_ID = "job-07b7991def2e396f507936e2a635b5de78a3f915"
LEGACY_SCHEMA = Path(__file__).absolute().parent / "fixtures" / "worker-v1.sql"


class ExecutionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="reclab-compatibility-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        source, public = self.root / "source", self.root / "public"
        for relative, raw in {
            "src/train.py": b"pass\n", "src/predict.py": b"pass\n",
            "configs/default.json": b"{}\n",
        }.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        public.mkdir()
        for name in ("manifest.json", "train.json", "valid.json"):
            (public / name).write_bytes(b"{}\n")
        self.source = execution.InputTree.freeze(source, "source")
        self.public = execution.InputTree.freeze(public, "public_data")
        self.policy = execution.WorkerPolicy(
            IMAGE, {"baseline": self.source}, {"tiny": self.public},
        )
        self.spec = execution.JobSpec.from_dict(json.loads(LEGACY_SPEC_JSON))
        self.state_root = self.root / "state"
        self.state_root.mkdir()

    def test_legacy_imports_reexport_the_same_types_and_functions(self):
        for name in (
            "CapabilityError", "CommandResult", "DockerError", "OwnershipError",
            "InputTree", "ResourceLimits", "JobSpec", "WorkerPolicy",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(execution, name), getattr(models, name))
        self.assertIs(execution.SubprocessDocker, docker.SubprocessDocker)
        self.assertIs(storage.confined_path, files.confined_path)
        self.assertIs(storage.atomic_write, files.atomic_write)
        self.assertIs(worker.load_policy, config.load_policy)

    def test_serialization_and_persisted_identity_match_legacy_vectors(self):
        self.assertEqual(canonical_json(asdict(self.spec)), LEGACY_SPEC_JSON)
        self.assertEqual(self.source.sha256, "2a9643a88b006fa7d705309db15e91c3806a64339b76c68872b0953f814d7e9b")
        self.assertEqual(self.public.sha256, "a6d569499b9041ca9054387cf630dbbc63a6c4203fac698557d77966520196f5")
        self.assertEqual(self.policy.identity(), LEGACY_POLICY_HASH)
        instance = execution.DockerWorker(self.state_root, self.policy, runner=FakeDocker())
        self.assertEqual(instance._key(self.spec), LEGACY_JOB_KEY)

    def test_legacy_database_and_receipt_open_without_relaunching_work(self):
        database = self.state_root / "jobs.sqlite3"
        with closing(sqlite3.connect(database)) as db, db:
            db.executescript(LEGACY_SCHEMA.read_text(encoding="utf-8"))
        output = self.state_root / "outputs" / LEGACY_JOB_ID
        output.mkdir(parents=True)
        (output / "model.json").write_bytes(b"{}\n")
        fake = FakeDocker()
        instance = execution.DockerWorker(
            self.state_root, self.policy, runner=fake, initialize=False,
        )
        self.assertEqual(instance.owner, "0123456789abcdef0123456789abcdef")
        saved = instance.get(LEGACY_JOB_ID)
        self.assertEqual(saved["job_key"], LEGACY_JOB_KEY)
        self.assertEqual(saved["policy_hash"], LEGACY_POLICY_HASH)
        self.assertEqual(saved["spec"], json.loads(LEGACY_SPEC_JSON))
        self.assertEqual(saved["receipt"]["sha256"], "44f2014af64f4344819e2da45413274dae5e685a0ced3f7102e3985bca936ac4")
        instance._verify_receipt(saved)
        self.assertEqual(instance.submit(self.spec), saved)
        restarted = execution.DockerWorker(self.state_root, self.policy, runner=fake)
        self.assertEqual(restarted.get(LEGACY_JOB_ID), saved)
        with closing(sqlite3.connect(database)) as db, db:
            self.assertEqual(db.execute("SELECT count(*) FROM docker_jobs").fetchone()[0], 1)
            self.assertEqual(
                db.execute("SELECT reason, confirmed FROM docker_stop_intents").fetchone(),
                ("cancelled", 1701.0),
            )
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
