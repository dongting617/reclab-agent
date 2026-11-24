"""Pipeline contract tests. Fixtures never execute model code or use SSH."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from reclab_agent.experiments.remote_runner import RemoteExperimentRunner
from reclab_agent.tools.contracts import ToolError
from tests import test_experiment_runner as fixtures


class FakeRemote:
    def __init__(self, config, predictions):
        self.connection = SimpleNamespace(host="fixture.invalid", port=22,
                                         remote_root="/fixture", python_executable="/python")
        self.config, self.predictions = config, predictions
        self.files, self.jobs, self.runs, self.stops = {}, {}, [], []
        self.fail_phase = None
        self.lost_submit = False
        self.confirm_stop = True
        self.mutate_inputs = False
        self.optional_artifacts_missing = False

    def bootstrap(self):
        return {"python": "fixture", "tmux": "fixture"}

    def upload(self, source_root, source, destination):
        local = (self.config.output_directory / "snapshots" if source_root == "source" else
                 self.config.dataset_directory / "public") / source
        self.files[destination] = local.read_bytes()
        return self.file_stat(destination)

    def file_stat(self, path):
        raw = self.files[path]
        return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def run_remote(self, job_id, argv, cwd, timeout_seconds, seed):
        self.runs.append((job_id, argv, timeout_seconds, seed))
        phase = "train" if job_id.endswith("-train") else "predict"
        job = {"job_id": job_id, "status": "failed" if phase == self.fail_phase else "completed",
               "exit_code": 7 if phase == self.fail_phase else 0, "stop_confirmed": True}
        self.jobs[job_id] = job
        if self.lost_submit:
            raise OSError("fixture SSH reply lost after remote start")
        if self.mutate_inputs:
            self.files[next(iter(self.files))] = b"changed by fixture candidate"
        return job

    def job_status(self, job_id):
        return self.jobs[job_id]

    def tail_log(self, job_id, lines):
        return {"text": "fixture log", "job_id": job_id}

    def stop_job(self, job_id):
        self.stops.append(job_id)
        return {"job_id": job_id, "stop_confirmed": self.confirm_stop, "status": "cancelled"}

    def download(self, source, destination):
        if self.optional_artifacts_missing and not source.endswith("predictions.json"):
            raise ToolError("NOT_FOUND", "fixture optional file missing")
        raw = json.dumps(self.predictions).encode() if source.endswith("predictions.json") else b"{}"
        local = self.config.output_directory / destination
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(raw)
        return {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class RemoteRunnerTests(unittest.TestCase):
    def setUp(self):
        fixtures.ExperimentRunnerTests.setUp(self)
        self.config = replace(self.config, execution_backend="ssh_tmux", execution_policy_path=None)
        self.remote = FakeRemote(self.config, self.predictions)

    def snapshot(self):
        snapshot = fixtures.ExperimentRunnerTests.snapshot(self)
        directory = self.config.output_directory / "snapshots" / snapshot["source_hash"]
        shutil.copytree(snapshot["directory"], directory, dirs_exist_ok=True)
        return {**snapshot, "directory": str(directory)}

    def runner(self):
        return RemoteExperimentRunner(self.config, self.plan, self.remote, registry=self.registry,
                                      clock=self.clock, sleep=self.clock.sleep)

    def execute(self, runner=None, snapshot=None):
        return (runner or self.runner())(snapshot or self.snapshot(), candidate_id="candidate-1",
                                         seed=42, max_seconds=10)

    def test_pipeline_scores_locally_and_has_native_identity(self):
        record = self.execute()
        self.assertEqual(record.status, "completed", record.failure)
        self.assertIsNone(record.image)
        self.assertEqual(record.runtime["backend"], "ssh_tmux")
        self.assertEqual(len(self.remote.runs), 2)
        self.assertEqual([run[2] for run in self.remote.runs], [8, 2])
        self.assertTrue(all("private" not in path for path in self.remote.files))
        grade = json.loads((self.config.output_directory / record.score.source).read_text())
        self.assertEqual(grade["metrics"]["ndcg@20"], 1)
        self.assertIn("--model", self.remote.runs[1][1])

    def test_completed_retry_uses_record_without_resubmitting(self):
        snapshot, runner = self.snapshot(), self.runner()
        first = self.execute(runner, snapshot)
        second = self.execute(runner, snapshot)
        self.assertEqual(first, second)
        self.assertEqual(len(self.remote.runs), 2)

    def test_resume_reuses_uploaded_inputs_and_existing_jobs_after_local_failure(self):
        snapshot = self.snapshot()
        first = self.execute(snapshot=snapshot)
        failed = first.to_dict()
        failed.update(status="failed", score=None, failure="upload: temporary SSH failure")
        record_path = self.config.output_directory / "experiments" / first.experiment_id / "record.json"
        record_path.write_text(json.dumps(failed))
        marker = self.config.output_directory / "active-remote-job.json"
        marker.write_text(json.dumps({"experiment_id": first.experiment_id, "job_id": first.training_job_id,
                                     "remote_root": "/fixture", "host": "fixture.invalid", "port": 22}))
        self.config = replace(self.config, resume_jobs=True)
        def no_upload(*args, **kwargs):
            raise AssertionError("recovery must not rewrite inputs used by the live job")
        self.remote.upload = no_upload
        result = self.execute(snapshot=snapshot)
        self.assertEqual(result.status, "completed", result.failure)
        self.assertEqual(result.training_job_id, first.training_job_id)
        self.assertEqual(len(self.remote.runs), 2)
        self.assertFalse(marker.exists())
        self.assertTrue((record_path.parent / "recovery-previous-record.json").exists())

    def test_resume_rejects_a_pending_job_from_another_experiment(self):
        snapshot = self.snapshot()
        marker = self.config.output_directory / "active-remote-job.json"
        marker.write_text(json.dumps({"experiment_id": "another-exp", "job_id": "another-exp-train",
                                     "remote_root": "/fixture", "host": "fixture.invalid", "port": 22}))
        self.config = replace(self.config, resume_jobs=True)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.execute(snapshot=snapshot)
        self.assertEqual(self.remote.runs, [])
        self.assertTrue(marker.exists())

    def test_prediction_recovery_accepts_explicit_unknown_job_without_retraining(self):
        snapshot = self.snapshot()
        first = self.execute(snapshot=snapshot)
        failed = first.to_dict()
        failed.update(status="failed", score=None, failure="prediction not submitted")
        (self.config.output_directory / "experiments" / first.experiment_id / "record.json").write_text(json.dumps(failed))
        marker = self.config.output_directory / "active-remote-job.json"
        marker.write_text(json.dumps({"experiment_id": first.experiment_id, "job_id": first.prediction_job_id,
                                     "remote_root": "/fixture", "host": "fixture.invalid", "port": 22}))
        self.config = replace(self.config, resume_jobs=True)
        del self.remote.jobs[first.prediction_job_id]
        self.remote.runs.clear()
        original_status = self.remote.job_status
        def status(job_id):
            if job_id not in self.remote.jobs:
                raise ToolError("INVALID_ARGUMENTS", "unknown job_id")
            return original_status(job_id)
        self.remote.job_status = status
        result = self.execute(snapshot=snapshot)
        self.assertEqual(result.status, "completed", result.failure)
        self.assertEqual([row[0] for row in self.remote.runs], [first.prediction_job_id])
        self.assertFalse(marker.exists())

    def test_training_failure_stops_without_prediction_or_score(self):
        self.remote.fail_phase = "train"
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertIsNone(record.score)
        self.assertEqual(len(self.remote.runs), 1)
        self.assertEqual(self.remote.stops, [record.training_job_id])

    def test_uncertain_submission_fences_next_experiment(self):
        self.remote.lost_submit, self.remote.confirm_stop = True, False
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertTrue((self.config.output_directory / "execution-blocked.json").exists())
        with self.assertRaisesRegex(RuntimeError, "unconfirmed"):
            self.execute()
        self.assertEqual(len(self.remote.runs), 1)

    def test_remote_input_mutation_prevents_scoring(self):
        self.remote.mutate_inputs = True
        record = self.execute()
        self.assertEqual(record.status, "failed")
        self.assertIn("inputs changed", record.failure)
        self.assertIsNone(record.score)

    def test_wrong_profile_rejected_without_bootstrap(self):
        with self.assertRaisesRegex(ValueError, "explicit ssh_tmux"):
            RemoteExperimentRunner(replace(self.config, execution_backend="docker"), self.plan,
                                   self.remote, registry=self.registry)

    def test_optional_training_metadata_is_not_part_of_model_contract(self):
        self.remote.optional_artifacts_missing = True
        self.assertEqual(self.execute().status, "completed")

    def test_restart_stops_prior_phase_before_new_upload_or_submission(self):
        snapshot = self.snapshot()
        marker = self.config.output_directory / "active-remote-job.json"
        marker.write_text(json.dumps({"experiment_id": "old-exp", "job_id": "old-live-job",
                                     "remote_root": "/fixture", "host": "fixture.invalid", "port": 22}))
        self.assertEqual(self.execute(snapshot=snapshot).status, "completed")
        self.assertEqual(self.remote.stops[0], "old-live-job")
        self.assertFalse(marker.exists())

    def test_restart_with_unconfirmed_prior_stop_blocks_new_training(self):
        snapshot = self.snapshot()
        marker = self.config.output_directory / "active-remote-job.json"
        marker.write_text(json.dumps({"experiment_id": "old-exp", "job_id": "old-live-job",
                                     "remote_root": "/fixture", "host": "fixture.invalid", "port": 22}))
        self.remote.confirm_stop = False
        with self.assertRaisesRegex(RuntimeError, "cannot be reconciled"):
            self.execute(snapshot=snapshot)
        self.assertEqual(self.remote.runs, [])
        self.assertTrue(marker.exists())


if __name__ == "__main__":
    unittest.main()
