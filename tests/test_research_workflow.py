"""Four-stage integration using real tools and a synthetic scorer, no GPU/API."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from reclab_agent.experiments.baselines import checkpoint_manifest
from reclab_agent.tools.contracts import ToolError
from tests import test_launch as launch_fixture


class WorkflowTests(unittest.TestCase):
    setUp = launch_fixture.LaunchTests.setUp
    binder = launch_fixture.LaunchTests.binder
    run_launch = launch_fixture.LaunchTests.run_launch

    def test_four_stage_order_and_hypothesis_precedes_candidate(self):
        report = self.run_launch()
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertEqual([x["stage"] for x in report["stages"]],
                         ["background", "hypothesis", "experiment", "analysis"])
        self.assertEqual(len(report["hypotheses"]), 1)
        research = self.research_instances[0]
        research.active_stage = "background"
        with self.assertRaises(ToolError):
            research.apply_patch("src/model.py", "width", "size", "0" * 64)
        with self.assertRaises(ToolError):
            research.run_experiment("candidate-1", 42)

    def test_explicit_baseline_reuses_checkpoint_with_one_training_slot(self):
        first = self.run_launch()
        root = Path(first["session_directory"])
        baseline = first["experiments"][0]
        checkpoint = root / "experiments" / baseline["record"]["experiment_id"] / "train/best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"synthetic checkpoint; never loaded as a model")
        first["baseline_artifacts"] = checkpoint_manifest(root, baseline)
        (root / "report.json").write_text(json.dumps(first), encoding="utf-8")
        self.config = replace(self.config, baseline_report=root / "report.json",
                              task=replace(self.config.task, budget=replace(self.config.task.budget, max_training_jobs=1)))
        self.records.clear()
        report = self.run_launch()
        self.assertEqual(report["status"], "completed", report.get("error"))
        self.assertTrue(report["baseline_reused"])
        self.assertEqual(report["budget_usage"]["experiment_attempts"], 1)
        self.assertEqual([x.candidate_id for x in self.records], ["candidate-1"])

    def test_unverified_history_stops_before_new_training(self):
        first = self.run_launch()
        self.config = replace(self.config, baseline_report=Path(first["report_path"]))
        self.records.clear()
        report = self.run_launch()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.records, [])

    def test_resume_reconciles_completed_results_without_resetting_budget(self):
        import reclab_agent.application.launch as launch
        first = self.run_launch()
        path = Path(first["report_path"])
        first["status"] = "failed"  # Simulate failure after scoring but before final delivery.
        first["error"] = "synthetic previous failure"
        path.write_text(json.dumps(first), encoding="utf-8")
        self.records.clear()
        resumed = launch.run_optimization(self.config, approve_plan=True, registry=self.registry,
            binder=self.binder, backend=launch_fixture.ScriptedLaunchBackend(), resume_directory=path.parent)
        self.assertEqual(resumed["status"], "completed", resumed.get("error"))
        self.assertNotIn("error", resumed)
        self.assertEqual(resumed["previous_failures"][-1]["error"], "synthetic previous failure")
        self.assertEqual(self.records, [])
        self.assertEqual(resumed["budget_usage"]["experiment_attempts"], first["budget_usage"]["experiment_attempts"])
        self.assertEqual(resumed["budget_usage"]["model_requests"], first["budget_usage"]["model_requests"])

    def test_missing_required_process_evidence_cannot_be_presented_as_improvement(self):
        class RequiredBackend(launch_fixture.ScriptedLaunchBackend):
            def call(self, name, arguments, suffix=None):
                if name == "record_hypothesis":
                    arguments["process_metrics"] = [{"name": "gradient_norm", "purpose": "required",
                        "expected": "finite and bounded", "operator": "<=", "threshold": 5.0}]
                return super().call(name, arguments, suffix)
        report = self.run_launch(RequiredBackend())
        self.assertFalse(report["comparison"][0]["improves"])
        self.assertEqual(report["comparison"][0]["decision"], "inconclusive")
        self.assertEqual(report["best_experiment_id"], report["experiments"][0]["record"]["experiment_id"])
        self.assertIn("证据不足", Path(report["user_report_markdown_path"]).read_text(encoding="utf-8"))

    def test_resume_pending_submission_does_not_reserve_a_second_training(self):
        import reclab_agent.application.launch as launch
        first = self.run_launch()
        root = Path(first["session_directory"])
        state = json.loads((root / "service-state.json").read_text())
        candidate = next(v for v in state["records"] if v["candidate_id"] == "candidate-1")
        state["records"] = [v for v in state["records"] if v["candidate_id"] != "candidate-1"]
        state["completed"] = [v for v in state["completed"] if v[1] != "candidate-1"]
        state["pending"] = {"request": [candidate["source_hash"], "candidate-1", 42], "seconds": 10}
        state["source_hash"] = candidate["source_hash"]
        (root / "service-state.json").write_text(json.dumps(state), encoding="utf-8")
        first.update(status="failed", comparison=[], experiments=first["experiments"][:1])
        (root / "report.json").write_text(json.dumps(first), encoding="utf-8")
        self.records.clear()
        result = launch.run_optimization(self.config, approve_plan=True, registry=self.registry,
            binder=self.binder, backend=launch_fixture.ScriptedLaunchBackend(), resume_directory=root)
        self.assertEqual(result["status"], "completed", result.get("error"))
        self.assertEqual(result["budget_usage"]["experiment_attempts"], 2)
        self.assertEqual(result["budget_usage"]["reserved_training_seconds"], first["budget_usage"]["reserved_training_seconds"])

    def test_bound_tool_catalog_matches_registered_capabilities(self):
        from reclab_agent.tools import tool_catalog
        self.run_launch()
        toolbox = self.research_instances[0]
        self.assertEqual({s.name for s in toolbox.specs},
                         {c.name for c in tool_catalog(toolbox=toolbox, availability="available")})

    def test_unconfirmed_remote_marker_preserves_pending_request_and_budget(self):
        original_binder = self.binder
        def binder(config, **kwargs):
            research = original_binder(config, **kwargs)
            runner = research._runner
            def uncertain(snapshot, **arguments):
                record = runner(snapshot, **arguments)
                (config.output_directory / "active-remote-job.json").write_text(
                    json.dumps({"experiment_id": record.experiment_id}))
                return replace(record, status="failed", score=None, failure="connection lost")
            research._runner = uncertain
            return research
        self.binder = binder
        report = self.run_launch()
        research = self.research_instances[0]
        self.assertEqual(report["status"], "failed")
        self.assertIsNotNone(research.pending_request)
        self.assertEqual(research.usage["experiment_attempts"], 1)
        self.assertEqual(research.experiments, [])

    def test_next_task_uses_the_verified_accepted_source_as_baseline(self):
        original_binder = self.binder
        def binder(config, **kwargs):
            research = original_binder(config, **kwargs)
            runner = research._runner
            def checkpoint_runner(snapshot, **arguments):
                record = runner(snapshot, **arguments)
                path = config.output_directory / "experiments" / record.experiment_id / "train/best.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"trusted synthetic fixture checkpoint")
                return record
            research._runner = checkpoint_runner
            return research
        self.binder = binder
        first = self.run_launch()
        self.assertNotIn("archive_error", first)
        self.config = replace(self.config, task=replace(self.config.task,
            budget=replace(self.config.task.budget, max_training_jobs=1)))
        class FurtherBackend(launch_fixture.ScriptedLaunchBackend):
            def call(self, name, arguments, suffix=None):
                if name == "apply_patch":
                    arguments.update(old="    width = 8\n", new="    width = 16\n")
                return super().call(name, arguments, suffix)
        second = self.run_launch(FurtherBackend())
        self.assertEqual(second["status"], "completed", {"error": second.get("error"),
            "reused": second.get("baseline_reused"), "stages": second.get("stages"), "budget": second.get("budget_usage")})
        self.assertNotIn("archive_error", second)
        self.assertTrue(second["baseline_reused"])
        self.assertEqual(second["budget_usage"]["experiment_attempts"], 1)
        self.assertEqual(second["experiments"][0]["record"]["source_hash"], first["experiments"][1]["record"]["source_hash"])
