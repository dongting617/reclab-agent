"""Suspended-controller recovery contracts; fixtures never use SSH or model code."""
from __future__ import annotations
import copy
from dataclasses import replace
import json
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.strategy_comparison import runner as experiment
from reclab_agent.domain.contracts import content_hash
from reclab_agent.domain.research import PlanReview
from reclab_agent.application.launch import LaunchError
from reclab_agent.experiments.remote_runner import RemoteExperimentRunner
from tests import test_remote_runner as fixtures


class CompletedRemote(fixtures.FakeRemote):
    """Model the helper's stable job IDs and independently recorded server times."""

    def __init__(self, config, predictions, clock):
        super().__init__(config, predictions)
        self.clock = clock
        self.executed = []

    def run_remote(self, job_id, argv, cwd, timeout_seconds, seed):
        if job_id in self.jobs:
            self.runs.append((job_id, argv, timeout_seconds, seed))
            return self.jobs[job_id]
        result = super().run_remote(job_id, argv, cwd, timeout_seconds, seed)
        self.executed.append(job_id)
        self.clock.now += 0.25
        result.update(started_at=100.0, finished_at=103.0 if job_id.endswith("-train") else 102.0)
        return result


class ExperimentRunnerRecoveryTests(unittest.TestCase):
    def setUp(self):
        fixtures.RemoteRunnerTests.setUp(self)
        self.config = replace(self.config, output_directory=self.root / "sc" / "session" / "baseline",
                              candidate_ids=(self.config.task.baseline_id, *self.config.candidate_ids))
        self.plan = replace(self.plan, candidate_ids=self.config.candidate_ids)
        self.remote = CompletedRemote(self.config, self.predictions, self.clock)

    def snapshot(self):
        return fixtures.RemoteRunnerTests.snapshot(self)

    def runner(self):
        return experiment.ExperimentRemoteRunner(self.config, self.plan, self.remote,
                                                 registry=self.registry, clock=self.clock,
                                                 sleep=self.clock.sleep)

    def prepare_reference(self):
        """Save an authentic fixture record outside the new controller's output."""
        snapshot = self.snapshot()
        prepared = replace(self.config, output_directory=self.root / "prepared")
        old_snapshot = prepared.output_directory / "snapshots" / snapshot["source_hash"]
        shutil.copytree(snapshot["directory"], old_snapshot)
        self.remote.config = prepared
        core = RemoteExperimentRunner(prepared, self.plan, self.remote, registry=self.registry,
                                      clock=self.clock, sleep=self.clock.sleep)
        record = core({**snapshot, "directory": str(old_snapshot)},
                      candidate_id=self.config.task.baseline_id, seed=42, max_seconds=10)
        self.assertEqual(record.status, "completed", record.failure)
        old = replace(record, status="timed_out", score=None, prediction_job_id=None,
                      failure="fixture controller suspended after successful training").to_dict()
        record_path = prepared.output_directory / "experiments" / record.experiment_id / "record.json"
        record_path.write_text(json.dumps(old), encoding="utf-8")
        reference = {"record_path": str(record_path), "reason": "fixture completed training recovery"}
        reference_path = self.root / "sc" / "baseline-reference.json"
        reference_path.write_text(json.dumps(reference), encoding="utf-8")
        self.remote.config = self.config
        self.remote.jobs.pop(record.prediction_job_id)
        self.remote.runs.clear()
        self.remote.executed.clear()
        return snapshot, old, record_path, reference

    def test_binder_uses_cpu_clock_preserves_remote_limits_and_records_all_timings(self):
        review = PlanReview(self.plan.plan_hash, "approved", "synthetic-test", "2026-11-20T00:00:00Z")
        wall = iter((100.0, 6700.0))
        clocks = SimpleNamespace(process_time=self.clock, monotonic=lambda: next(wall))
        with patch.object(experiment, "time", clocks):
            research = experiment.bind_experiment_tools(self.config, plan=self.plan, review=review,
                                                        registry=self.registry, remote_tools=self.remote)
            self.assertIs(research._runner.clock, self.clock)
            record = research._runner(self.snapshot(), candidate_id="candidate-1", seed=42, max_seconds=10)
        self.assertEqual(record.status, "completed", record.failure)
        self.assertEqual([run[2] for run in self.remote.runs], [8, 2])
        path = self.config.output_directory / "experiments" / record.experiment_id / "execution.json"
        execution = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(execution["remote_phase_timeout_seconds"], {"train": 8, "predict": 2})
        self.assertTrue(execution["per_phase_timeout_enforced_remotely"])
        self.assertEqual(execution["elapsed_seconds"], 6600.0)
        self.assertEqual(execution["controller_cpu_seconds"], 0.5)
        self.assertEqual(execution["server_job_seconds"], 5.0)
        self.assertEqual(execution["supplemental_controller_clock"], "process_cpu_excluding_desktop_suspension")

    def test_mismatched_reference_identity_never_submits_remote_work(self):
        snapshot, old, record_path, _ = self.prepare_reference()
        replacements = {
            "experiment_id": "different-experiment",
            "plan_hash": "a" * 64,
            "source_hash": "b" * 64,
            "config_hash": "c" * 64,
            "dataset_id": "different-dataset",
            "protocol_hash": "d" * 64,
            "candidate_id": "different-candidate",
            "seed": 43,
            "training_job_id": "different-training-job",
        }
        runtime = copy.deepcopy(old["runtime"])
        runtime["details"]["host"] = "another-fixture.invalid"
        runtime["fingerprint"] = content_hash(runtime["details"])
        replacements["runtime"] = runtime
        invalid_runtime = copy.deepcopy(old["runtime"])
        invalid_runtime["fingerprint"] = "f" * 64
        replacements["invalid-runtime"] = invalid_runtime
        runner = self.runner()
        for field, replacement in replacements.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(old)
                changed["runtime" if field == "invalid-runtime" else field] = replacement
                record_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(LaunchError):
                    runner(snapshot, candidate_id=self.config.task.baseline_id, seed=42, max_seconds=10)
                self.assertEqual(self.remote.runs, [])
                self.assertEqual(self.remote.executed, [])

    def test_completed_baseline_reuses_training_job_and_continues_to_prediction(self):
        snapshot, old, _, reference = self.prepare_reference()
        record = self.runner()(snapshot, candidate_id=self.config.task.baseline_id, seed=42, max_seconds=10)
        self.assertEqual(record.status, "completed", record.failure)
        self.assertEqual(record.experiment_id, old["experiment_id"])
        self.assertEqual(record.training_job_id, old["training_job_id"])
        self.assertEqual([run[0] for run in self.remote.runs],
                         [old["training_job_id"], record.prediction_job_id])
        self.assertEqual(self.remote.executed, [record.prediction_job_id])
        self.assertIn("--model", self.remote.runs[-1][1])
        self.assertIsNotNone(record.score)
        path = self.config.output_directory / "experiments" / record.experiment_id / "execution.json"
        execution = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(execution["reused_baseline_training"], reference)


if __name__ == "__main__":
    unittest.main()
