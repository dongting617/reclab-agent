"""Strategy comparison contracts with synthetic scores, never API/GPU execution."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from reclab_agent.contracts import Budget, canonical_json, content_hash
from reclab_agent.core import ModelTurn
from experiments.strategy_comparison import controller as comparison
from experiments.strategy_comparison.settings import ComparisonConfiguration, with_comparison
from reclab_agent.research.contracts import EvidenceRef, ExperimentRecord
from reclab_agent.research.launch import LaunchError
from reclab_agent.research.toolbox import ResearchToolbox
from reclab_agent.workspace import create_workspace
from tests import test_experiment_runner as fixtures


class FixtureComparisonBackend:
    def __init__(self, invalid_probe=False):
        self.probes = self.calls = 0
        self.invalid_probe = invalid_probe

    def respond(self, *, instructions, input_items, tools):
        self.calls += 1
        if any(tool.get("name") == "startup_check" for tool in tools):
            self.probes += 1
            if self.invalid_probe:
                return ModelTurn([], "No function-call support")
            return ModelTurn([{"type": "function_call", "name": "startup_check", "arguments": "{}",
                               "call_id": "fixture-probe"}], "", usage={"input_tokens": 1, "output_tokens": 1,
                                                                          "total_tokens": 2})
        return ModelTurn([], "Fixture model observation", usage={"input_tokens": 2, "output_tokens": 3,
                                                                  "total_tokens": 5})


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        fixtures.ExperimentRunnerTests.setUp(self)
        self.strategies = ("react", "plan_execute", "reflection")
        self.configure(2)
        self.records = []
        self.bindings = []
        self.agent_observations = []
        self.runner_observations = []
        self.checkpoint_observations = []
        self.no_candidate = False
        self.corrupt_strategy = None
        self.block_strategy = None
        self.block_marker = "execution-blocked.json"
        self.model_calls_per_round = 1
        self.environment = patch.dict("os.environ", {
            self.config.model_env: "fixture-model", self.config.api_key_env: "credential-never-returned",
            "OPENAI_BASE_URL": "https://fixture.invalid/v1",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def configure(self, count):
        self.count = count
        budget = Budget(max_training_jobs=1 + len(self.strategies) * count,
                        max_model_turns=1 + len(self.strategies) * 20,
                        max_patches=len(self.strategies) * 4, max_job_seconds=10,
                        max_total_job_seconds=(1 + len(self.strategies) * count) * 10)
        self.config = with_comparison(self.config, ComparisonConfiguration(self.strategies, count,
                                      max_plan_steps=3, max_reflections=1),
                                      task=replace(self.config.task, budget=budget),
                                      candidate_ids=tuple(f"candidate-{index}" for index in range(1, count + 1)))

    def binder(self, config, *, plan, review, registry, remote_tools=None):
        workspace = create_workspace(config.output_directory / "workspaces", "fixture-candidate",
                                     self.files, editable_paths=list(config.editable_files))
        binding = {"config": config, "plan": plan, "workspace": workspace}
        self.bindings.append(binding)

        def run(snapshot, *, candidate_id, seed, max_seconds):
            mode = {"single": "react", "reviewer": "reflection"}.get(config.task.mode, config.task.mode)
            baseline = candidate_id == config.task.baseline_id
            text = (Path(snapshot["directory"]) / "src/model.py").read_text(encoding="utf-8")
            round_number = 0 if baseline else int(candidate_id.rsplit("-", 1)[1])
            observed = {"strategy": None if baseline else mode, "baseline": baseline,
                        "candidate_id": candidate_id, "round": round_number,
                        "source_hash": snapshot["source_hash"], "source_text": text,
                        "output": str(config.output_directory), "max_seconds": max_seconds}
            self.runner_observations.append(observed)
            experiment_id = "fixture-" + content_hash([plan.plan_hash, snapshot["source_hash"],
                                                         candidate_id, seed])[:24]
            identity = dict(experiment_id=experiment_id, plan_hash=plan.plan_hash, candidate_id=candidate_id,
                            source_hash=snapshot["source_hash"], config_hash=content_hash(self.model_config),
                            dataset_id=plan.dataset_id, protocol_hash=plan.protocol_hash,
                            image=fixtures.FIXTURE_IMAGE, seed=seed, program_validation="passed")
            if not baseline and mode == self.block_strategy:
                fence = config.output_directory / self.block_marker
                fence.write_text(json.dumps({"job_id": experiment_id + "-train", "stop_confirmed": False}),
                                 encoding="utf-8")
                record = ExperimentRecord(**identity, status="failed", training_job_id=experiment_id + "-train",
                                          failure="fixture stopping remains unconfirmed")
            else:
                # Real fixed scorer, controlled ranking positions. No metric is
                # accepted from the Agent or from this fixture runner's claims.
                ranks = {"react": (1, 4), "plan_execute": (2, 1), "reflection": (4, 2)}
                rank = 3 if baseline else ranks[mode][min(round_number - 1, 1)]
                predictions = {}
                for user, ideal in self.predictions.items():
                    target = ideal[0]
                    ranking = list(ideal[1:])
                    ranking.insert(rank - 1, target)
                    predictions[user] = ranking
                grade = registry.score_predictions(config.evaluator_id, self.bundle, predictions)
                relative = "experiments/" + experiment_id + "/grade.json"
                path = config.output_directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(canonical_json(grade), encoding="utf-8")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if not baseline and mode == self.corrupt_strategy:
                    digest = "0" * 64
                score = EvidenceRef(relative, digest, "/metrics/ndcg@20", "Synthetic comparison test only")
                record = ExperimentRecord(**identity, status="completed", training_job_id=experiment_id + "-train",
                                          prediction_job_id=experiment_id + "-predict", score=score)
            self.records.append(record)
            return record

        research = ResearchToolbox(workspace, task=config.task, plan=plan, review=review,
                                   snapshots_root=config.output_directory / "snapshots", run_experiment=run)
        binding["research"] = research
        return research

    def agent_factory(self, research, backend, **options):
        owner = self
        mode = options.get("kind", research.task.mode)
        mode = {"single": "react", "reviewer": "reflection"}.get(mode, mode)
        spec = next(spec for spec in research.specs if spec.name == "run_experiment")
        candidate = spec.input_schema["properties"]["candidate_id"]["enum"][0]
        checkpoint_path = Path(research.snapshots_root).parent / "report.json"
        if checkpoint_path.exists():
            self.checkpoint_observations.append(json.loads(checkpoint_path.read_text(encoding="utf-8")))

        class FixtureAgent:
            def run(self, prompt):
                before = research.invoke("read_source", {"path": "src/model.py"})["result"]
                owner.agent_observations.append({"strategy": mode, "candidate_id": candidate,
                    "prompt": prompt, "source_text": before["text"], "source_hash": before["sha256"],
                    "workspace": str(research.workspace.root), "options": options})
                for _ in range(owner.model_calls_per_round):
                    backend.respond(instructions="Fixture decision", input_items=[{"role": "user", "content": "fixture"}],
                                    tools=[])
                if not owner.no_candidate:
                    patch = research.invoke("apply_patch", {"path": "src/model.py", "old": before["text"],
                        "new": before["text"] + f"# {mode}:{candidate}\n", "expected_sha256": before["sha256"]})
                    if not patch["ok"]:
                        raise AssertionError(patch)
                    result = research.invoke("run_experiment", {"candidate_id": candidate, "seed": 42})
                    if not result["ok"]:
                        raise AssertionError(result)
                return {"status": "completed", "answer": "Untrusted claim: NDCG@20 improved to 999.",
                        "steps": 1, "tool_calls": 3}
        return FixtureAgent()

    def run_comparison(self, backend=None, **kwargs):
        return comparison.run_comparison(self.config, approve_plan=True,
            backend=backend or FixtureComparisonBackend(), registry=self.registry,
            binder=self.binder, agent_factory=self.agent_factory, **kwargs)

    def test_unapproved_comparison_never_probes_or_binds(self):
        backend = FixtureComparisonBackend()
        with self.assertRaises(LaunchError):
            comparison.run_comparison(self.config, backend=backend, registry=self.registry,
                                      binder=self.binder, agent_factory=self.agent_factory)
        self.assertEqual(backend.calls, 0)
        self.assertEqual(self.bindings, [])

    def test_api_compatibility_failure_precedes_baseline_or_candidate_jobs(self):
        backend = FixtureComparisonBackend(invalid_probe=True)
        with self.assertRaises(LaunchError):
            self.run_comparison(backend)
        self.assertEqual(backend.probes, 1)
        self.assertEqual(self.records, [])

    def test_one_baseline_and_equal_isolated_strategy_allocations(self):
        backend = FixtureComparisonBackend()
        report = self.run_comparison(backend)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(sum(item["baseline"] for item in self.runner_observations), 1)
        self.assertEqual(len(self.records), 1 + 3 * self.count)
        self.assertEqual(backend.probes, 1)
        strategies = report["strategies"]
        self.assertEqual({row["strategy"] for row in strategies}, set(self.strategies))
        self.assertTrue(all(len(row["rounds"]) == self.count for row in strategies))
        self.assertTrue(all(row["budget_usage"]["experiment_attempts"] == self.count for row in strategies))
        budgets = [entry["config"].task.budget for entry in self.bindings[1:]]
        self.assertEqual(len(budgets), 3)
        self.assertEqual(len({entry["plan"].plan_hash for entry in self.bindings[1:]}), 3)
        self.assertTrue(all(budget == budgets[0] for budget in budgets))
        self.assertEqual(budgets[0].max_training_jobs, self.count)
        self.assertEqual(budgets[0].max_model_turns, 20)
        self.assertEqual(budgets[0].max_total_job_seconds, 10 * self.count)
        first_rounds = [row for row in self.agent_observations if row["candidate_id"] == "candidate-1"]
        self.assertEqual(len({row["workspace"] for row in first_rounds}), 3)
        self.assertTrue(all(row["source_text"] == self.files["src/model.py"] for row in first_rounds))
        self.assertEqual((self.source / "src/model.py").read_text(encoding="utf-8"), self.files["src/model.py"])

    def test_strategy_inherits_its_verified_best_and_discards_regressions(self):
        report = self.run_comparison()
        observations = {(row["strategy"], row["candidate_id"]): row for row in self.agent_observations}
        self.assertIn("react:candidate-1", observations[("react", "candidate-2")]["source_text"])
        self.assertNotIn("reflection:candidate-1", observations[("reflection", "candidate-2")]["source_text"])
        strategies = {row["strategy"]: row for row in report["strategies"]}
        self.assertEqual(strategies["react"]["best"]["record"]["candidate_id"], "candidate-1")
        self.assertEqual(strategies["plan_execute"]["best"]["record"]["candidate_id"], "candidate-2")
        self.assertEqual(strategies["reflection"]["best"]["record"]["candidate_id"], "candidate-2")
        for name, entry in strategies.items():
            best_dir = Path(entry["best"]["source_directory"])
            text = (best_dir / "src/model.py").read_text(encoding="utf-8")
            self.assertNotIn("react:candidate-2", text)
            if name != "react":
                self.assertNotIn("react:candidate-1", text)

    def test_histories_do_not_leak_other_strategy_experiment_results(self):
        self.run_comparison()
        react_ids = [record.experiment_id for record in self.records if any(
            row["strategy"] == "react" and row["source_hash"] == record.source_hash
            for row in self.runner_observations)]
        later_prompts = [row["prompt"] for row in self.agent_observations if row["strategy"] != "react"]
        self.assertTrue(react_ids)
        self.assertTrue(all(identity not in prompt for prompt in later_prompts for identity in react_ids))

    def test_independent_scores_override_agent_quality_claims(self):
        report = self.run_comparison()
        self.assertTrue(all(row["best_metrics"]["ndcg@20"] <= 1 for row in report["comparison"]))
        self.assertEqual(set(report["winning_strategies"]), {"react", "plan_execute"})
        self.assertNotIn("credential-never-returned", json.dumps(report))
        saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "completed")

    def test_strategy_specific_options_and_model_budgets_remain_equal(self):
        report = self.run_comparison()
        options = {row["strategy"]: row["options"] for row in self.agent_observations}
        self.assertEqual(options["plan_execute"]["max_plan_steps"], 3)
        self.assertEqual(options["reflection"]["max_reflections"], 1)
        self.assertNotIn("max_reflections", options["react"])
        self.assertNotIn("max_plan_steps", options["react"])
        self.assertEqual({row["model_calls"] for row in report["strategies"]}, {self.count})
        self.assertEqual(report["budget_usage"]["experiment_attempts"], 1 + 3 * self.count)

    def test_round_checkpoints_are_public_progress_without_controller_backreferences(self):
        self.run_comparison()
        self.assertEqual(len(self.checkpoint_observations), 3)
        for checkpoint in self.checkpoint_observations:
            self.assertEqual(len(checkpoint["rounds"]), 1)
            self.assertNotIn("global_report", checkpoint)
            self.assertNotIn("global_output", checkpoint)
            self.assertNotIn("windows_ssh", checkpoint)

    def test_no_measured_candidates_is_incomplete_without_a_winner(self):
        self.no_candidate = True
        report = self.run_comparison()
        self.assertEqual(report["status"], "incomplete")
        self.assertFalse(report["winning_strategies"])
        self.assertEqual(len(self.records), 1)

    def test_corrupt_score_hash_prevents_promotion(self):
        self.corrupt_strategy = "react"
        report = self.run_comparison()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["winning_strategies"])
        self.assertTrue(all(row.get("best", {}).get("record", {}).get("candidate_id") != "candidate-1"
                            for row in report["strategies"]))

    def _assert_unconfirmed_stop_prevents_later_strategies_and_restart(self):
        self.block_strategy = "react"
        report = self.run_comparison()
        self.assertEqual(report["status"], "failed")
        self.assertTrue((self.config.output_directory / "comparison-active.json").exists())
        bound_modes = {{"single": "react", "reviewer": "reflection"}.get(
            binding["config"].task.mode, binding["config"].task.mode) for binding in self.bindings}
        self.assertNotIn("plan_execute", bound_modes)
        self.assertNotIn("reflection", bound_modes)
        before = len(self.records)
        with self.assertRaises(LaunchError):
            self.run_comparison()
        self.assertEqual(len(self.records), before)

    def test_unconfirmed_stop_prevents_later_strategies_and_restart(self):
        self._assert_unconfirmed_stop_prevents_later_strategies_and_restart()

    def test_active_job_marker_also_blocks_later_strategies(self):
        self.block_marker = "active-remote-job.json"
        self._assert_unconfirmed_stop_prevents_later_strategies_and_restart()

    def test_model_request_budget_is_not_reset_between_rounds(self):
        self.model_calls_per_round = 15
        backend = FixtureComparisonBackend()
        report = self.run_comparison(backend)
        self.assertLessEqual(backend.calls, self.config.task.budget.max_model_turns)
        self.assertEqual(len(report["strategies"]), 3)
        self.assertEqual({row["model_calls"] for row in report["strategies"]}, {20})
        self.assertFalse(report["winning_strategies"])

    def test_new_completed_comparisons_use_distinct_output_sessions(self):
        self.configure(1)
        first = self.run_comparison()
        second = self.run_comparison()
        self.assertNotEqual(first["session_directory"], second["session_directory"])
        self.assertNotEqual(first["report_path"], second["report_path"])
        self.assertTrue(Path(first["report_path"]).is_file())
        self.assertTrue(Path(second["report_path"]).is_file())
        self.assertFalse((self.config.output_directory / "comparison-active.json").exists())

    def assert_user_reports(self, report):
        html = Path(report["user_report_path"])
        markdown = Path(report["user_report_markdown_path"])
        self.assertTrue(html.is_file())
        self.assertTrue(markdown.is_file())
        self.assertEqual(html.parent, Path(report["session_directory"]))
        self.assertIn("ndcg@20", html.read_text(encoding="utf-8"))
        self.assertIn("ndcg@20", markdown.read_text(encoding="utf-8"))
        self.assertTrue(report["user_summary"])
        self.assertNotIn("Untrusted claim", report["user_summary"])
        saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["user_report_path"], str(html))

    def test_completed_comparison_automatically_writes_user_reports(self):
        report = self.run_comparison()
        self.assert_user_reports(report)
        self.assertEqual(report["context"]["target_model"], self.model_config["model"]["class"])
        self.assertEqual(report["context"]["guardrail_metrics"], list(self.config.guardrail_metrics))

    def test_incomplete_comparison_still_writes_user_reports(self):
        self.no_candidate = True
        report = self.run_comparison()
        self.assertEqual(report["status"], "incomplete")
        self.assert_user_reports(report)
        self.assertFalse(report["winning_strategies"])

    def test_failed_score_keeps_attempt_evidence_and_user_reports(self):
        self.corrupt_strategy = "react"
        report = self.run_comparison()
        self.assertEqual(report["status"], "failed")
        self.assert_user_reports(report)
        attempted = report["strategies"][0]["rounds"][0]["experiments"]
        self.assertEqual(len(attempted), 1)
        self.assertNotIn("metrics", attempted[0])

    def test_cancelled_comparison_writes_user_reports_before_reraising(self):
        with patch.object(self, "agent_factory", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_comparison()
        session, = (self.config.output_directory / "sessions").iterdir()
        report = json.loads((session / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "cancelled")
        self.assert_user_reports(report)
        self.assertFalse((self.config.output_directory / "comparison-active.json").exists())

    def test_report_export_error_preserves_machine_report_and_safe_cleanup(self):
        with patch.object(comparison, "write_user_report", side_effect=OSError("private fixture detail")):
            report = self.run_comparison()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["user_report_error"], "OSError")
        self.assertNotIn("private fixture detail", json.dumps(report))
        self.assertTrue(Path(report["report_path"]).is_file())
        self.assertFalse((self.config.output_directory / "comparison-active.json").exists())


if __name__ == "__main__":
    unittest.main()
