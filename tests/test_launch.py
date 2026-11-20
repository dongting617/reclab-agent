"""Real launch control flow with synthetic data; no API or GPU calls."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from reclab_agent.contracts import canonical_json, content_hash
from reclab_agent.core import ModelTurn
from reclab_agent.research import launch
from reclab_agent.research.contracts import EvidenceRef, ExperimentRecord
from reclab_agent.research.toolbox import ResearchToolbox
from reclab_agent.workspace import create_workspace
from tests import test_experiment_runner as fixtures


class ScriptedLaunchBackend:
    """Produce controlled function calls through the actual Agent tool loop."""

    def __init__(self, *, no_candidate=False, invalid_probe=False, extra_experiment=False):
        self.no_candidate = no_candidate
        self.invalid_probe = invalid_probe
        self.extra_experiment = extra_experiment
        self.steps = 0
        self.probes = 0
        self.tool_outputs = []

    def respond(self, *, instructions, input_items, tools):
        names = {tool.get("name") for tool in tools}
        if "startup_check" in names:
            self.probes += 1
            if self.invalid_probe:
                return ModelTurn([], "Text completion without compatible tool output")
            return self.call("startup_check", {}, "probe")
        if self.no_candidate:
            return ModelTurn([], "completed: candidate quality is excellent")
        self.steps += 1
        if input_items and "output" in input_items[-1]:
            self.tool_outputs.append(json.loads(input_items[-1]["output"]))
        if self.steps == 1:
            return self.call("read_source", {"path": "src/model.py"})
        if self.steps == 2:
            source = self.tool_outputs[-1]["result"]
            return self.call("apply_patch", {"path": "src/model.py", "old": "    pass\n",
                                             "new": "    width = 8\n",
                                             "expected_sha256": source["sha256"]})
        if self.steps == 3:
            return self.call("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
        if self.extra_experiment:
            if self.steps == 4:
                return self.call("read_source", {"path": "src/model.py"})
            if self.steps == 5:
                source = self.tool_outputs[-1]["result"]
                return self.call("apply_patch", {"path": "src/model.py", "old": "    width = 8\n",
                                                 "new": "    width = 16\n",
                                                 "expected_sha256": source["sha256"]})
            if self.steps == 6:
                return self.call("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
        return ModelTurn([], "Synthetic Agent finished; compare independent scores.")

    def call(self, name, arguments, suffix=None):
        return ModelTurn([{"type": "function_call", "name": name,
                           "arguments": json.dumps(arguments),
                           "call_id": "launch-fixture-" + (suffix or str(self.steps))}], "")


class LaunchTests(unittest.TestCase):
    def setUp(self):
        fixtures.ExperimentRunnerTests.setUp(self)
        self.records = []
        self.research_instances = []
        self.bound_configurations = []
        self.corrupt_grade = False
        self.failed_baseline = False
        self.environment = patch.dict("os.environ", {
            self.config.model_env: "fixture-model", self.config.api_key_env: "credential-never-returned",
            "OPENAI_BASE_URL": "https://fixture.invalid/v1",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def binder(self, config, *, plan, review, registry, remote_tools=None):
        self.bound_configurations.append(config)
        files = {name: text for name, text in self.files.items()}
        workspace = create_workspace(config.output_directory / "workspaces", "fixture-candidate",
                                     files, editable_paths=list(config.editable_files))

        def run(snapshot, *, candidate_id, seed, max_seconds):
            experiment_id = "fixture-" + content_hash([snapshot["source_hash"], candidate_id, seed])[:24]
            identity = dict(experiment_id=experiment_id, plan_hash=plan.plan_hash, candidate_id=candidate_id,
                            source_hash=snapshot["source_hash"], config_hash=content_hash(self.model_config),
                            dataset_id=plan.dataset_id, protocol_hash=plan.protocol_hash,
                            image=fixtures.FIXTURE_IMAGE, seed=seed, program_validation="passed")
            if self.failed_baseline and candidate_id == config.task.baseline_id:
                record = ExperimentRecord(**identity, status="failed", failure="fixture train failed")
            else:
                predictions = {}
                for user, rankings in self.predictions.items():
                    ranking = list(rankings)
                    if candidate_id == config.task.baseline_id:
                        ranking[0], ranking[1] = ranking[1], ranking[0]
                    predictions[user] = ranking
                grade = registry.score_predictions(config.evaluator_id, self.bundle, predictions)
                relative = "experiments/" + experiment_id + "/grade.json"
                path = config.output_directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(canonical_json(grade), encoding="utf-8")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if self.corrupt_grade:
                    digest = "0" * 64
                score = EvidenceRef(relative, digest, "/metrics/ndcg@20", "Synthetic launch test only")
                record = ExperimentRecord(**identity, status="completed", training_job_id=experiment_id + "-train",
                                          prediction_job_id=experiment_id + "-predict", score=score)
            self.records.append(record)
            return record

        research = ResearchToolbox(workspace, task=config.task, plan=plan, review=review,
                                   snapshots_root=config.output_directory / "snapshots", run_experiment=run)
        self.research_instances.append(research)
        return research

    def run_launch(self, backend=None, **kwargs):
        return launch.run_optimization(self.config, approve_plan=True,
                                       backend=backend or ScriptedLaunchBackend(),
                                       registry=self.registry, binder=self.binder, **kwargs)

    def assert_user_report(self, report):
        self.assertTrue(Path(report["user_report_path"]).is_file())
        self.assertTrue(Path(report["user_report_markdown_path"]).is_file())
        self.assertTrue(report["user_summary"])
        self.assertNotIn("credential-never-returned",
                         Path(report["user_report_path"]).read_text(encoding="utf-8"))
        saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["user_report_path"], report["user_report_path"])

    def test_unapproved_launch_never_calls_provider_or_binds_execution(self):
        backend = ScriptedLaunchBackend()
        with self.assertRaises(launch.LaunchError):
            launch.run_optimization(self.config, backend=backend, registry=self.registry, binder=self.binder)
        self.assertEqual(backend.probes, 0)
        self.assertEqual(self.research_instances, [])
        self.assertFalse(self.config.output_directory.exists())

    def test_prepare_is_readonly_and_includes_baseline_with_one_shared_seed(self):
        config, plan, inspection = launch.prepare_launch(replace(self.config, seeds=(42, 43)),
                                                        registry=self.registry)
        self.assertTrue(inspection["inputs_valid"])
        self.assertIn(self.config.task.baseline_id, config.candidate_ids)
        self.assertIn("candidate-1", plan.candidate_ids)
        self.assertEqual(plan.seeds, (42,))
        self.assertFalse(config.output_directory.exists())

    def test_invalid_tool_probe_fails_before_any_gpu_binding(self):
        backend = ScriptedLaunchBackend(invalid_probe=True)
        with self.assertRaises(launch.LaunchError):
            self.run_launch(backend)
        self.assertEqual(backend.probes, 1)
        self.assertEqual(self.research_instances, [])
        self.assertEqual(self.records, [])

    def test_baseline_and_candidate_share_budget_and_independent_scores(self):
        report = self.run_launch()
        self.assert_user_report(report)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(len(self.research_instances), 1)
        self.assertEqual(report["budget_usage"]["experiment_attempts"], 2)
        self.assertEqual([record.candidate_id for record in self.records],
                         [self.config.task.baseline_id, "candidate-1"])
        self.assertEqual(len(report["experiments"]), 2)
        self.assertTrue(report["comparison"][0]["improves"])
        self.assertTrue(report["comparison"][0]["passes_guardrails"])
        review = self.research_instances[0].review
        self.assertEqual(review.plan_hash, self.research_instances[0].plan.plan_hash)
        self.assertEqual(review.decision, "approved")
        self.assertNotIn("credential-never-returned", json.dumps(report))
        saved = Path(report["report_path"]).read_text(encoding="utf-8")
        self.assertNotIn("credential-never-returned", saved)
        self.assertEqual(json.loads(saved)["status"], "completed")

    def test_agent_completion_without_candidate_is_incomplete(self):
        report = self.run_launch(ScriptedLaunchBackend(no_candidate=True))
        self.assert_user_report(report)
        self.assertNotIn("candidate quality is excellent", report["user_summary"])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(len(self.records), 1)
        self.assertEqual(report["budget_usage"]["experiment_attempts"], 1)
        self.assertFalse(report["comparison"])

    def test_extra_attempt_cannot_reset_baseline_budget(self):
        backend = ScriptedLaunchBackend(extra_experiment=True)
        report = self.run_launch(backend)
        self.assertEqual(report["budget_usage"]["experiment_attempts"], 2)
        self.assertEqual(len(self.records), 2)
        errors = [output.get("error", {}).get("code") for output in backend.tool_outputs if not output["ok"]]
        self.assertIn("BUDGET_EXCEEDED", errors)

    def test_corrupt_grade_never_counts_as_verified_improvement(self):
        self.corrupt_grade = True
        report = self.run_launch()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["comparison"])
        self.assert_user_report(report)
        self.assertTrue(all("metrics" not in entry for entry in report["experiments"]))

    def test_failed_baseline_stops_before_candidate_agent(self):
        self.failed_baseline = True
        backend = ScriptedLaunchBackend()
        report = self.run_launch(backend)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(self.records), 1)
        self.assertEqual(backend.steps, 0)
        self.assertEqual(report["budget_usage"]["experiment_attempts"], 1)
        self.assertFalse(report["comparison"])
        self.assert_user_report(report)

    def test_agent_failure_preserves_verified_baseline_and_writes_user_report(self):
        agent = Mock()
        agent.run.side_effect = RuntimeError("credential-never-returned")
        report = self.run_launch(agent_factory=Mock(return_value=agent))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], "RuntimeError")
        self.assertEqual(len(report["experiments"]), 1)
        self.assertIn("metrics", report["experiments"][0])
        self.assert_user_report(report)

    def test_interrupted_agent_writes_cancelled_user_report_before_reraising(self):
        agent = Mock()
        agent.run.side_effect = KeyboardInterrupt()
        progress = []
        with self.assertRaises(KeyboardInterrupt):
            self.run_launch(agent_factory=Mock(return_value=agent), progress=progress.append)
        output = self.bound_configurations[-1].output_directory
        saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "cancelled")
        self.assertIn("metrics", saved["experiments"][0])
        self.assert_user_report(saved)
        self.assertTrue(progress[-1].startswith("User report: "))

    def test_user_report_write_failure_keeps_machine_evidence_and_measured_status(self):
        with patch.object(launch, "write_user_report", side_effect=OSError("credential-never-returned")):
            report = self.run_launch()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["user_report_error"], "OSError")
        self.assertEqual(len(report["experiments"]), 2)
        saved = Path(report["report_path"]).read_text(encoding="utf-8")
        self.assertNotIn("credential-never-returned", saved)
        self.assertEqual(json.loads(saved)["user_report_error"], "OSError")

    def test_two_explicit_runs_get_distinct_output_sessions(self):
        first = self.run_launch()
        second = self.run_launch()
        self.assertNotEqual(first["session_directory"], second["session_directory"])
        self.assertNotEqual(first["report_path"], second["report_path"])
        self.assertTrue(Path(first["report_path"]).is_file())
        self.assertTrue(Path(second["report_path"]).is_file())
        self.assertEqual([research.usage["experiment_attempts"] for research in self.research_instances], [2, 2])

    def test_api_failure_does_not_reflect_credentials(self):
        backend = Mock()
        backend.respond.side_effect = RuntimeError("upstream reflected credential-never-returned")
        with self.assertRaises(launch.LaunchError) as raised:
            launch.check_agent_api(self.config, backend=backend)
        self.assertNotIn("credential-never-returned", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
