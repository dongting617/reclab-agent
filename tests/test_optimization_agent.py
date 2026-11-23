"""Development-loop contracts using scripted decisions, never real training."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from reclab_agent import ModelTurn
from reclab_agent.domain.contracts import Budget, TaskSpec
from reclab_agent.domain.research import EvidenceRef, ExperimentRecord, PlanReview, ResearchPlan
from reclab_agent.agents.optimization import OptimizationAgent
from reclab_agent.tools.toolbox import OptimizationToolbox
from reclab_agent.experiments.workspace import create_workspace
from reclab_agent.tools import RecLabToolbox


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        # The Windows app sandbox's default temp root is already very long;
        # content-addressed snapshots add a 64-character directory component.
        directory = tempfile.TemporaryDirectory(prefix="reclab-opt-", dir=Path(__file__).absolute().parent)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.files = {"src/train.py": "learning_rate = 0.1\n", "src/predict.py": "pass\n",
                      "configs/default.json": "{}\n"}
        self.workspace = create_workspace(self.root, "candidate", self.files)
        self.budget = Budget(max_model_turns=6, max_patches=2, max_training_jobs=2,
                             max_job_seconds=10, max_total_job_seconds=20)
        self.task = TaskSpec("task-1", "Test a learning-rate change on development data", "tiny", "baseline",
                             budget=self.budget)
        self.plan = ResearchPlan(
            plan_id="plan-1", task_id=self.task.task_id, hypothesis="A lower learning rate may help",
            dataset_id="tiny", data_profile="synthetic-contract-fixture", baseline_id="baseline",
            protocol_hash="a" * 64, primary_metric="ndcg@20", editable_files=tuple(self.files),
            comparison="same data and seed", expected_observation="compare independent development scores",
            rejection_criteria="reject if score decreases", stop_conditions=("budget exhausted",),
            budget=self.budget, seeds=(42,),
        )
        self.review = PlanReview(self.plan.plan_hash, "approved", "user", "2026-11-19T12:00:00+08:00")
        self.runner_calls = []

    def fixture_runner(self, snapshot, *, candidate_id, seed, max_seconds):
        """Trusted test fixture: validates metadata, does not run candidate code."""
        self.runner_calls.append((snapshot, candidate_id, seed, max_seconds))
        self.assertTrue(Path(snapshot["directory"], "src/train.py").is_file())
        return ExperimentRecord(
            experiment_id=f"fixture-{len(self.runner_calls)}", plan_hash=self.plan.plan_hash,
            candidate_id=candidate_id, source_hash=snapshot["source_hash"], config_hash="b" * 64,
            dataset_id=self.plan.dataset_id, protocol_hash=self.plan.protocol_hash,
            image="example/fixture@sha256:" + "c" * 64, seed=seed,
            status="completed", program_validation="passed", training_job_id="fixture-train",
            prediction_job_id="fixture-predict", score=EvidenceRef(
                "reports/fixture-score.json", "d" * 64, "/metrics/ndcg@20", "synthetic fixture; not a model-quality claim"),
        )

    def toolbox(self, *, review=True, runner=True, **kwargs):
        return OptimizationToolbox(self.workspace, task=self.task, plan=self.plan,
                               review=self.review if review is True else review,
                               snapshots_root=self.root / "snapshots",
                               run_experiment=self.fixture_runner if runner is True else runner, **kwargs)

    def test_scripted_agent_reads_patches_validates_and_runs_frozen_source(self):
        class ScriptedBackend:
            def __init__(backend_self):
                backend_self.calls = 0

            def respond(backend_self, *, instructions, input_items, tools):
                backend_self.calls += 1
                if backend_self.calls == 1:
                    name, arguments = "read_source", {"path": "src/train.py"}
                elif backend_self.calls == 2:
                    source = json.loads(input_items[-1]["output"])["result"]
                    name, arguments = "apply_patch", {
                        "path": "src/train.py", "old": "0.1", "new": "0.01",
                        "expected_sha256": source["sha256"],
                    }
                elif backend_self.calls == 3:
                    name, arguments = "validate_candidate", {}
                elif backend_self.calls == 4:
                    name, arguments = "run_experiment", {"candidate_id": "candidate-1", "seed": 42}
                else:
                    return ModelTurn(output=[], output_text="The development fixture completed; no quality claim.")
                return ModelTurn(output=[{"type": "function_call", "name": name,
                                           "arguments": json.dumps(arguments),
                                           "call_id": f"fixture-call-{backend_self.calls}"}], output_text="")

        toolbox = self.toolbox()
        agent = OptimizationAgent(toolbox, ScriptedBackend())
        result = agent.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["task_id"], self.task.task_id)
        self.assertEqual(result["optimization_scope"], "development_only")
        self.assertEqual(result["budget_usage"]["patch_attempts"], 1)
        self.assertEqual(result["budget_usage"]["experiment_attempts"], 1)
        snapshot, candidate_id, seed, max_seconds = self.runner_calls[0]
        self.assertEqual(Path(snapshot["directory"], "src/train.py").read_text(), "learning_rate = 0.01\n")
        self.assertEqual((candidate_id, seed, max_seconds), ("candidate-1", 42, 10))
        record = result["experiments"][0]
        self.assertEqual(record["source_hash"], snapshot["source_hash"])
        self.assertEqual(record["score"]["source"], "reports/fixture-score.json")
        self.assertIn("does not establish model improvement", result["quality_claim"])

    def test_reads_work_before_approval_but_edits_and_runs_do_not(self):
        toolbox = self.toolbox(review=None)
        read = toolbox.invoke("read_source", {"path": "src/train.py"})
        self.assertTrue(read["ok"])
        self.assertTrue(toolbox.invoke("validate_candidate")["ok"])
        patch = toolbox.invoke("apply_patch", {"path": "src/train.py", "old": "0.1", "new": "0.01",
                                                "expected_sha256": read["result"]["sha256"]})
        run = toolbox.invoke("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
        self.assertEqual(patch["error"]["code"], "PLAN_NOT_APPROVED")
        self.assertEqual(run["error"]["code"], "PLAN_NOT_APPROVED")
        self.assertEqual(self.runner_calls, [])

    def test_model_context_describes_scope_without_granting_unreviewed_execution(self):
        self.plan = replace(self.plan, candidate_ids=("candidate-lr",), seeds=(7,),
                            guardrail_metrics=("recall@20",))
        first_prompts, rejected = [], []

        class ContextBackend:
            def __init__(self):
                self.calls = 0

            def respond(self, *, instructions, input_items, tools):
                self.calls += 1
                if self.calls == 1:
                    first_prompts.append(input_items[0]["content"])
                    name, arguments = "read_source", {"path": "src/train.py"}
                elif self.calls == 2:
                    source = json.loads(input_items[-1]["output"])["result"]
                    name, arguments = "apply_patch", {
                        "path": source["path"], "old": "0.1", "new": "0.01",
                        "expected_sha256": source["sha256"],
                    }
                elif self.calls == 3:
                    rejected.append(json.loads(input_items[-1]["output"])["error"]["code"])
                    name, arguments = "run_experiment", {"candidate_id": "candidate-lr", "seed": 7}
                else:
                    rejected.append(json.loads(input_items[-1]["output"])["error"]["code"])
                    return ModelTurn([], "The registered scope still requires user approval.")
                return ModelTurn([{"type": "function_call", "name": name,
                                   "arguments": json.dumps(arguments),
                                   "call_id": f"context-{self.calls}"}], "")

        toolbox = self.toolbox(review=None)
        request = "Inspect the registered SASRec development candidate."
        result = OptimizationAgent(toolbox, ContextBackend(), target_model="SASRec").run(request)
        first_prompt = first_prompts[0]
        context, _ = json.JSONDecoder().raw_decode(first_prompt[first_prompt.index("{"):])
        self.assertIn(request, first_prompt)
        self.assertIn(self.task.question, first_prompt)
        self.assertEqual(context["target_model"], "SASRec")
        self.assertEqual(context["task"]["task_id"], self.task.task_id)
        self.assertEqual(context["task"]["dataset_id"], self.task.dataset_id)
        self.assertEqual(context["task"]["baseline_id"], self.task.baseline_id)
        scope = context["plan"]
        self.assertEqual(set(scope["editable_files"]), set(self.plan.editable_files))
        self.assertEqual(scope["candidate_ids"], ["candidate-lr"])
        self.assertEqual(scope["seeds"], [7])
        self.assertEqual(scope["protocol_hash"], self.plan.protocol_hash)
        self.assertEqual(scope["plan_hash"], self.plan.plan_hash)
        self.assertEqual(scope["primary_metric"], self.plan.primary_metric)
        self.assertEqual(scope["guardrail_metrics"], ["recall@20"])
        self.assertEqual(scope["budget"], self.plan.to_dict()["budget"])
        self.assertEqual(rejected, ["PLAN_NOT_APPROVED", "PLAN_NOT_APPROVED"])
        self.assertEqual(self.workspace.read("src/train.py")["text"], self.files["src/train.py"])
        self.assertEqual(self.runner_calls, [])
        self.assertEqual(result["budget_usage"]["patch_attempts"], 0)
        self.assertEqual(result["budget_usage"]["experiment_attempts"], 0)

    def test_independent_score_artifact_becomes_readable_model_feedback(self):
        """Exercise real artifact reads; the metric is explicitly a fixture."""
        def runner(snapshot, **kwargs):
            record = self.fixture_runner(snapshot, **kwargs)
            raw = json.dumps({"metrics": {"ndcg@20": 0.25}, "split": "valid"}).encode()
            score_path = self.root / "reports" / "fixture-score.json"
            score_path.parent.mkdir()
            score_path.write_bytes(raw)
            return replace(record, score=EvidenceRef(
                "reports/fixture-score.json", hashlib.sha256(raw).hexdigest(),
                "/metrics/ndcg@20", "independent fixture, not a real trained model",
            ))

        observed = []

        class FeedbackBackend:
            def __init__(self):
                self.calls = 0

            def respond(self, *, instructions, input_items, tools):
                self.calls += 1
                if self.calls == 1:
                    name, args = "run_experiment", {"candidate_id": "candidate-1", "seed": 42}
                elif self.calls == 2:
                    score = json.loads(input_items[-1]["output"])["result"]["score"]
                    name, args = "inspect_artifact", {"path": score["source"],
                                                      "expected_sha256": score["sha256"]}
                elif self.calls == 3:
                    identity = json.loads(input_items[-1]["output"])["result"]
                    observed.append(identity["sha256_matches"])
                    name, args = "read_json_values", {"path": identity["path"],
                                                     "pointers": ["/metrics/ndcg@20"]}
                else:
                    values = json.loads(input_items[-1]["output"])["result"]["values"]
                    observed.append(values["/metrics/ndcg@20"])
                    return ModelTurn([], "The fixture metric is 0.25; see reports/fixture-score.json")
                return ModelTurn([{"type": "function_call", "name": name,
                                   "arguments": json.dumps(args), "call_id": str(self.calls)}], "")

        agent = OptimizationAgent(self.toolbox(runner=runner), FeedbackBackend(),
                                  evidence=RecLabToolbox(self.root))
        result = agent.run()
        self.assertEqual((result["status"], result["tool_calls"]), ("completed", 3))
        self.assertEqual(observed, [True, 0.25])
        self.assertIn("reports/fixture-score.json", result["evidence_paths"])

    def test_workspace_must_not_add_unreviewed_files_to_snapshot(self):
        self.plan = replace(self.plan, editable_files=("src/train.py",))
        with self.assertRaisesRegex(ValueError, "exactly match"):
            self.toolbox()

    def test_outside_plan_candidates_seeds_and_stale_review_are_rejected(self):
        toolbox = self.toolbox()
        for arguments in ({"candidate_id": "candidate-2", "seed": 42},
                          {"candidate_id": "candidate-1", "seed": 43}):
            self.assertEqual(toolbox.invoke("run_experiment", arguments)["error"]["code"], "OUTSIDE_PLAN")
        self.plan = replace(self.plan, revision=2)
        stale = self.toolbox()
        self.assertEqual(stale.invoke("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
                         ["error"]["code"], "PLAN_NOT_APPROVED")
        self.assertEqual(self.runner_calls, [])

    def test_syntax_failure_never_reaches_runner(self):
        toolbox = self.toolbox()
        source = toolbox.read_source("src/train.py")
        toolbox.apply_patch("src/train.py", source["text"], "def broken(:\n", source["sha256"])
        result = toolbox.invoke("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
        self.assertEqual(result["error"]["code"], "CANDIDATE_INVALID")
        self.assertEqual(toolbox.usage["experiment_attempts"], 0)
        self.assertEqual(self.runner_calls, [])

    def test_runner_failure_consumes_attempt_budget(self):
        self.plan = replace(self.plan, budget=replace(self.budget, max_training_jobs=1))
        self.review = replace(self.review, plan_hash=self.plan.plan_hash)
        calls = []

        def failing_runner(snapshot, **kwargs):
            calls.append(snapshot)
            raise RuntimeError("fixture execution failed")

        toolbox = self.toolbox(runner=failing_runner)
        arguments = {"candidate_id": "candidate-1", "seed": 42}
        self.assertEqual(toolbox.invoke("run_experiment", arguments)["error"]["code"], "EXPERIMENT_RUN_FAILED")
        self.assertEqual(toolbox.invoke("run_experiment", arguments)["error"]["code"], "BUDGET_EXCEEDED")
        self.assertEqual(len(calls), 1)
        self.assertEqual(toolbox.usage["experiment_attempts"], 1)

    def test_record_must_match_snapshot_and_evaluation_identity(self):
        def mismatched_runner(snapshot, **kwargs):
            return replace(self.fixture_runner(snapshot, **kwargs), source_hash="0" * 64)

        toolbox = self.toolbox(runner=mismatched_runner)
        result = toolbox.invoke("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
        self.assertEqual(result["error"]["code"], "INVALID_EXPERIMENT_RESULT")
        self.assertEqual(toolbox.experiments, [])
        self.assertEqual(toolbox.usage["experiment_attempts"], 1)

    def test_runner_cannot_change_expected_identity_by_mutating_snapshot_metadata(self):
        def mutating_runner(snapshot, **kwargs):
            snapshot["source_hash"] = "0" * 64
            return self.fixture_runner(snapshot, **kwargs)

        toolbox = self.toolbox(runner=mutating_runner)
        result = toolbox.invoke("run_experiment", {"candidate_id": "candidate-1", "seed": 42})
        self.assertEqual(result["error"]["code"], "INVALID_EXPERIMENT_RESULT")

    def test_plan_model_turn_budget_caps_whole_agent_run(self):
        self.plan = replace(self.plan, budget=replace(self.budget, max_model_turns=2))
        self.review = replace(self.review, plan_hash=self.plan.plan_hash)

        class AlwaysReadBackend:
            calls = 0

            def respond(backend_self, **kwargs):
                backend_self.calls += 1
                return ModelTurn(output=[{"type": "function_call", "name": "read_source",
                                           "arguments": '{"path":"src/train.py"}',
                                           "call_id": f"read-{backend_self.calls}"}], output_text="")

        backend = AlwaysReadBackend()
        result = OptimizationAgent(self.toolbox(), backend, max_steps=32).run()
        self.assertEqual(result["status"], "step_limit_exceeded")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(backend.calls, 2)

    def test_duplicate_request_uses_existing_record_without_spending_again(self):
        toolbox = self.toolbox()
        arguments = {"candidate_id": "candidate-1", "seed": 42}
        first = toolbox.invoke("run_experiment", arguments)
        second = toolbox.invoke("run_experiment", arguments)
        self.assertEqual(first, second)
        self.assertEqual(len(self.runner_calls), 1)
        self.assertEqual(toolbox.get_experiment("fixture-1"), first["result"])

    def test_runner_capabilities_are_absent_when_not_configured(self):
        toolbox = self.toolbox(runner=None)
        self.assertEqual({spec.name for spec in toolbox.specs},
                         {"read_source", "apply_patch", "validate_candidate", "record_hypothesis", "read_log", "write_diagnostic", "run_diagnostics", "resolve_baseline", "analyze_experiment"})
        self.assertEqual(toolbox.invoke("run_experiment")["error"]["code"], "UNKNOWN_TOOL")


if __name__ == "__main__":
    unittest.main()
