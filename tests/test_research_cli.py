"""Controlled demo tests; mocked jobs do not establish Docker isolation."""
import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from reclab_agent import demo
from reclab_agent.contracts import content_hash
from reclab_agent.datasets import popularity_predictions
from reclab_agent.execution import CapabilityError, DockerWorker, InputTree, JobSpec
from reclab_agent.research_cli import main, run_supervised
from reclab_agent.workspace import Workspace
from tests.support.docker import WatchdogFakeDocker


IMAGE = "docker.io/library/python@sha256:" + "a" * 64


class CompletedWorker:
    def __init__(self, bundle):
        self.bundle, self.specs, self.collected = bundle, [], []

    def capabilities(self):
        return {"docker": True}

    def submit(self, spec):
        self.specs.append(spec)
        return {"job_id": "job-" + spec.phase, "status": "running"}

    def poll(self, job_id):
        return {"job_id": job_id, "status": "completed", "exit_code": 0,
                "receipt": {"sha256": content_hash({"job": job_id}), "files": {"fixture": {"bytes": 1}}}}

    def collect_predictions(self, job_id):
        self.collected.append(job_id)
        return popularity_predictions(self.bundle, self.bundle.evaluation_inputs("valid"))


class DemoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "controlled-demo"

    def prepare(self):
        return demo.prepare(self.root)

    def docker_fixture(self, *, exit_code=0, finish=True):
        self.prepare()
        policy, bundle, manifest = demo.policy_for(self.root, IMAGE)
        fake = WatchdogFakeDocker()
        fake.image_digest = IMAGE
        worker = DockerWorker(self.root / "worker", policy, runner=fake)
        job = worker.submit(JobSpec(demo.SOURCE_ID, bundle.dataset_id, limits=demo.LIMITS))
        if finish:
            (worker._output(job["job_id"]) / "counts.json").write_text("{}")
            fake.containers[job["container"]]["State"].update(Status="exited", ExitCode=exit_code)
            job = worker.poll(job["job_id"])
        demo._write(self.root, "run.json", {"status": job["status"], "image": IMAGE,
                                           "jobs": [demo._summary(job)]})
        return fake, worker, bundle, manifest, job

    def test_prepare_only_validates_syntax_and_freezes_separate_snapshot(self):
        sentinel = self.parent / "host-executed"
        files = {**demo.FILES, "src/train.py": f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"}
        with patch.dict(demo.FILES, files), patch("subprocess.run", side_effect=AssertionError("host subprocess")):
            manifest = self.prepare()
        self.assertFalse(sentinel.exists())
        self.assertIn("syntax_only", manifest["validation"]["checks"])
        snapshot = self.root / manifest["snapshot"]
        self.assertNotEqual(snapshot, self.root / "workspaces" / demo.SOURCE_ID)
        self.assertEqual(InputTree.freeze(snapshot, "source").sha256, manifest["source_tree_sha256"])

    def test_existing_root_is_never_overwritten(self):
        self.root.mkdir()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_snapshot_and_public_data_identities_are_verified(self):
        manifest = self.prepare()
        source = self.root / manifest["snapshot"] / "src/train.py"
        source.write_text("print('changed')", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unchanged audited"):
            demo.load(self.root)

    def test_workspace_changes_do_not_change_snapshot(self):
        manifest = self.prepare()
        workspace = Workspace(self.root / "workspaces" / demo.SOURCE_ID, list(demo.FILES))
        current = workspace.read("src/train.py")
        workspace.replace("src/train.py", "random.seed(seed)", "random.seed(seed + 1)", current["sha256"])
        self.assertEqual(demo.load(self.root)[1]["source_hash"], manifest["source_hash"])

    def test_only_public_tree_is_registered_test_labels_cannot_be_mounted(self):
        manifest = self.prepare()
        policy, bundle, _ = demo.policy_for(self.root, IMAGE)
        tree = policy.datasets[bundle.dataset_id]
        self.assertEqual(tree.directory, self.root / "dataset/public")
        self.assertFalse((tree.directory / "test.json").exists())
        self.assertTrue((self.root / "dataset/private/test.json").exists())
        with self.assertRaises(ValueError):
            InputTree.freeze(self.root / "dataset", "public_data")
        self.assertFalse(manifest["test_labels_mounted"])

    def test_mutable_image_rejected_without_binding_run(self):
        self.prepare()
        with self.assertRaises(ValueError):
            run_supervised(str(self.root), "python:latest")
        self.assertFalse((self.root / "run.json").exists())

    def test_train_predict_collection_and_independent_grade(self):
        self.prepare()
        _, _, bundle, _, _ = demo.load(self.root)
        worker = CompletedWorker(bundle)
        result = demo.execute(self.root, IMAGE, worker=worker, sleep=lambda _: None)
        self.assertEqual([s.phase for s in worker.specs], ["train", "predict"])
        self.assertEqual(worker.specs[1].model_job_id, "job-train")
        self.assertEqual([s.seed for s in worker.specs], [42, 42])
        self.assertEqual([asdict(s.limits) for s in worker.specs], [asdict(demo.LIMITS)] * 2)
        self.assertEqual(worker.collected, ["job-predict"])
        report = json.loads((self.root / "report.json").read_text())
        self.assertEqual(result["report_sha256"], content_hash(report))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["quality_evidence"], "fixture_only")
        self.assertFalse(result["llm_connected"])

    def test_failed_training_never_submits_prediction(self):
        self.prepare()
        worker = CompletedWorker(demo.load(self.root)[2])
        worker.poll = lambda job_id: {"job_id": job_id, "status": "failed"}
        with self.assertRaisesRegex(RuntimeError, "train job"):
            demo.execute(self.root, IMAGE, worker=worker, sleep=lambda _: None)
        self.assertEqual(len(worker.specs), 1)

    def test_missing_docker_is_structured_and_has_no_host_fallback(self):
        self.prepare()
        output = io.StringIO()
        with patch("reclab_agent.execution.DockerWorker.capabilities", side_effect=CapabilityError("no Docker")), contextlib.redirect_stdout(output):
            code = main(["_execute", "--root", str(self.root), "--image", IMAGE])
        value = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(value["error"], "CapabilityError")
        self.assertFalse(value["host_fallback"])
        self.assertFalse((self.root / "predictions.json").exists())

    def test_supervisor_hard_timeout_then_separate_stop_confirmation(self):
        self.prepare()
        calls = []
        def child(root, image, operation, timeout):
            calls.append((operation, timeout))
            if operation == "_execute":
                raise subprocess.TimeoutExpired(["trusted-controller"], timeout)
            return {"ok": True, "result": {"stop_confirmed": True, "jobs": []}}
        with patch("reclab_agent.research_cli._child", side_effect=child):
            result = run_supervised(str(self.root), IMAGE)
        self.assertEqual(calls, [("_execute", 120), ("_cancel", 30)])
        self.assertFalse(result["ok"])
        self.assertTrue(result["stop"]["stop_confirmed"])

    def test_cancellation_timeout_never_claims_stopped(self):
        self.prepare()
        with patch("reclab_agent.research_cli._child", side_effect=subprocess.TimeoutExpired(["trusted-controller"], 1)):
            result = run_supervised(str(self.root), IMAGE)
        self.assertFalse(result["stop"]["stop_confirmed"])
        self.assertIn("watchdog", result["operator_action"])

    def test_another_image_cannot_admit_more_jobs_on_same_root(self):
        self.prepare()
        worker = CompletedWorker(demo.load(self.root)[2])
        demo.execute(self.root, IMAGE, worker=worker, sleep=lambda _: None)
        with self.assertRaisesRegex(ValueError, "bound to one image"):
            run_supervised(str(self.root), IMAGE.replace("a" * 64, "b" * 64))
        self.assertEqual(len(worker.specs), 2)

    def test_completed_container_restart_requires_fresh_owned_exit_check(self):
        fake, worker, bundle, manifest, job = self.docker_fixture()
        self.assertEqual(job["status"], "completed")
        container = fake.containers[job["container"]]
        container["State"]["Status"] = "running"
        before = fake.count("inspect")
        with patch("reclab_agent.demo.worker_for", return_value=(worker, bundle, manifest)):
            result = demo.cancel_and_confirm(self.root, IMAGE)
        self.assertTrue(result["stop_confirmed"])
        self.assertGreater(fake.count("inspect"), before)
        self.assertEqual(container["State"]["Status"], "exited")
        self.assertIn(["container", "kill", "--signal", "KILL", "--", container["Id"]], fake.calls)
        self.assertEqual(result["confirmation_scope"], "observation_only")
        self.assertTrue(result["continued_watchdog_required"])
        with worker._connect() as db:
            self.assertEqual(db.execute("SELECT reason FROM docker_stop_intents WHERE job_id=?",
                                        (job["job_id"],)).fetchone()[0], "cancelled")
        self.assertEqual(demo.status(self.root)["run"]["status"], "cancelled")

    def test_failed_container_restart_is_not_trusted_as_terminal(self):
        fake, worker, bundle, manifest, job = self.docker_fixture(exit_code=1)
        self.assertEqual(job["status"], "failed")
        fake.containers[job["container"]]["State"]["Status"] = "running"
        with patch("reclab_agent.demo.worker_for", return_value=(worker, bundle, manifest)):
            result = demo.cancel_and_confirm(self.root, IMAGE)
        self.assertTrue(result["stop_confirmed"])
        self.assertEqual(fake.containers[job["container"]]["State"]["Status"], "exited")

    def test_replaced_ownership_never_signalled_or_marked_stopped(self):
        fake, worker, bundle, manifest, job = self.docker_fixture()
        container = fake.containers[job["container"]]
        container["Config"]["Labels"]["io.reclab.worker"] = "another-owner"
        container["State"]["Status"] = "running"
        before = fake.count("kill")
        with patch("reclab_agent.demo.worker_for", return_value=(worker, bundle, manifest)):
            result = demo.cancel_and_confirm(self.root, IMAGE)
        self.assertFalse(result["stop_confirmed"])
        self.assertEqual(fake.count("kill"), before)
        self.assertEqual(container["State"]["Status"], "running")

    def test_unconfirmed_kill_preserves_unconfirmed_state(self):
        fake, worker, bundle, manifest, job = self.docker_fixture(finish=False)
        fake.kill_confirmed = False
        with patch("reclab_agent.demo.worker_for", return_value=(worker, bundle, manifest)):
            result = demo.cancel_and_confirm(self.root, IMAGE)
        self.assertFalse(result["stop_confirmed"])
        self.assertEqual(demo.status(self.root)["run"]["status"], "stop_unconfirmed")
        self.assertEqual(fake.containers[job["container"]]["State"]["Status"], "running")

    def test_error_path_refreshes_job_rows_after_cancellation(self):
        fake, worker, bundle, manifest, job = self.docker_fixture(finish=False)
        def child(root, image, operation, timeout):
            if operation == "_execute":
                return {"ok": False, "error": "DockerError", "detail": "injected failure"}
            return {"ok": True, "result": demo.cancel_and_confirm(root, image)}
        with patch("reclab_agent.demo.worker_for", return_value=(worker, bundle, manifest)), \
                patch("reclab_agent.research_cli._child", side_effect=child):
            result = run_supervised(str(self.root), IMAGE)
        state = demo.status(self.root)["run"]
        self.assertFalse(result["ok"])
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["jobs"][0]["status"], "cancelled")
        self.assertEqual(state["jobs"][0]["stop_observation"]["container_state"], "exited")
        self.assertNotEqual(state["jobs"][0]["status"], "running")

    def test_status_explicitly_marks_cached_container_liveness(self):
        self.prepare()
        result = demo.status(self.root)
        self.assertFalse(result["live_verified"])
        self.assertEqual(result["container_liveness"], "unverified")
        self.assertIn("cached", result["state_source"])


if __name__ == "__main__":
    unittest.main()
