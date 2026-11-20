"""Comparison round controls with real workspaces and non-executing runners."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from reclab_agent.contracts import Budget, TaskSpec, content_hash
from reclab_agent.core import ModelTurn
from experiments.strategy_comparison.tools import (
    CountedBackend, ModelBudgetExceeded, RoundResearch, restore_snapshot,
)
from reclab_agent.research.contracts import EvidenceRef, ExperimentRecord, PlanReview, ResearchPlan
from reclab_agent.research.toolbox import ResearchToolbox
from reclab_agent.workspace import create_workspace


class FakeBackend:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.attempts = 0

    def respond(self, **arguments):
        self.attempts += 1
        value = next(self.turns)
        if isinstance(value, Exception):
            raise value
        return value


class CountedBackendTests(unittest.TestCase):
    def test_successes_stop_at_limit_without_another_provider_request(self):
        backend = FakeBackend([ModelTurn([], "one"), ModelTurn([], "two")])
        counted = CountedBackend(backend, 2)
        self.assertEqual(counted.calls, 0)
        counted.respond(instructions="one", input_items=[], tools=[])
        counted.respond(instructions="two", input_items=[], tools=[])
        with self.assertRaises(ModelBudgetExceeded):
            counted.respond(instructions="three", input_items=[], tools=[])
        self.assertEqual(counted.calls, 2)
        self.assertEqual(backend.attempts, 2)

    def test_failed_requests_also_consume_the_budget(self):
        backend = FakeBackend([RuntimeError("fixture-provider-secret")])
        counted = CountedBackend(backend, 1)
        with self.assertRaises(RuntimeError):
            counted.respond()
        with self.assertRaises(ModelBudgetExceeded) as failure:
            counted.respond()
        self.assertNotIn("fixture-provider-secret", str(failure.exception))
        self.assertEqual(counted.calls, 1)
        self.assertEqual(backend.attempts, 1)

    def test_usage_aggregates_only_nonnegative_json_integers(self):
        backend = FakeBackend([
            ModelTurn([], "", usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
            ModelTurn([], "", usage={"input_tokens": True, "output_tokens": -1, "total_tokens": 4.0}),
            ModelTurn([], "", usage={"input_tokens": 7, "output_tokens": "9"}),
        ])
        counted = CountedBackend(backend, 3)
        for _ in range(3):
            counted.respond()
        self.assertEqual(counted.usage, {"input_tokens": 10, "output_tokens": 2, "total_tokens": 5})
        detached = counted.usage
        detached["input_tokens"] = 999
        self.assertEqual(counted.usage["input_tokens"], 10)

    def test_invalid_limits_and_backends_fail_before_any_calls(self):
        backend = FakeBackend([])
        for limit in (0, -1, True, 1.0):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                CountedBackend(backend, limit)
        with self.assertRaises(TypeError):
            CountedBackend(object(), 1)


class RoundResearchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="comparison-tools-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = {"src/model.py": "class Model:\n    width = 1\n",
                      "src/dependency.py": "constant = 7\n",
                      "configs/default.json": '{"epochs":1}\n'}
        self.workspace = create_workspace(self.root, "source", self.files,
                                          editable_paths=["src/model.py", "configs/default.json"])
        self.budget = Budget(max_training_jobs=3, max_job_seconds=10, max_total_job_seconds=30)
        self.task = TaskSpec("comparison-fixture", "exercise comparison controls", "tiny", "baseline",
                             budget=self.budget)
        self.plan = ResearchPlan(
            "comparison-plan", self.task.task_id, "test candidate", self.task.dataset_id, "synthetic fixture",
            self.task.baseline_id, "b" * 64, "ndcg@20", tuple(sorted(self.workspace.editable_paths)),
            "fixed fixture", "independent fixture record", "reject malformed output", ("budget exhausted",),
            candidate_ids=("candidate-1", "candidate-2"), seeds=(42,), budget=self.budget,
        )
        review = PlanReview(self.plan.plan_hash, "approved", "synthetic-test", "2026-11-19T00:00:00Z")
        self.submitted = []
        self.fail_runner = False

        def runner(snapshot, *, candidate_id, seed, max_seconds):
            self.submitted.append((snapshot, candidate_id, seed, max_seconds))
            experiment_id = "fixture-" + content_hash([snapshot["source_hash"], candidate_id])[:24]
            record = ExperimentRecord(
                experiment_id, self.plan.plan_hash, candidate_id, snapshot["source_hash"], "c" * 64,
                self.task.dataset_id, self.plan.protocol_hash, "fixture/python@sha256:" + "a" * 64, seed,
                status="failed", program_validation="passed", failure="synthetic failure",
            )
            if self.fail_runner:
                return record
            return replace(record, status="completed", failure=None,
                training_job_id=experiment_id + "-train", prediction_job_id=experiment_id + "-predict",
                score=EvidenceRef("fixture-grade.json", "d" * 64, "/metrics/ndcg@20", "Synthetic only"))

        self.research = ResearchToolbox(self.workspace, task=self.task, plan=self.plan, review=review,
                                       snapshots_root=self.root / "snapshots", run_experiment=runner)
        self.baseline = self.workspace.snapshot(self.research.snapshots_root)

    def patch_model(self, width=2):
        current = self.research.read_source("src/model.py")
        old = current["text"]
        new = f"class Model:\n    width = {width}\n"
        return self.research.apply_patch("src/model.py", old, new, current["sha256"])

    def invoke(self, round_tools, candidate="candidate-1", seed=42):
        return round_tools.invoke("run_experiment", {"candidate_id": candidate, "seed": seed})

    def test_round_restricts_schema_without_mutating_the_underlying_tools(self):
        tools = RoundResearch(self.research, "candidate-1", 42, set())
        spec = next(spec for spec in tools.specs if spec.name == "run_experiment")
        self.assertEqual(spec.input_schema["properties"]["candidate_id"]["enum"], ["candidate-1"])
        self.assertEqual(spec.input_schema["properties"]["seed"]["enum"], [42])
        original = next(spec for spec in self.research.specs if spec.name == "run_experiment")
        self.assertNotIn("enum", original.input_schema["properties"]["candidate_id"])
        spec.input_schema["properties"]["candidate_id"]["enum"].append("wrong")
        fresh = next(spec for spec in tools.specs if spec.name == "run_experiment")
        self.assertEqual(fresh.input_schema["properties"]["candidate_id"]["enum"], ["candidate-1"])

    def test_round_preserves_source_reads_and_shared_budget_properties(self):
        tools = RoundResearch(self.research, "candidate-1", 42, set())
        self.assertIs(tools.task, self.research.task)
        self.assertIs(tools.plan, self.research.plan)
        self.assertIs(tools.workspace, self.workspace)
        self.assertTrue(tools.invoke("read_source", {"path": "src/dependency.py"})["ok"])
        self.assertEqual(tools.usage, self.research.usage)

    def test_wrong_candidate_seed_or_argument_shape_never_submits(self):
        tools = RoundResearch(self.research, "candidate-1", 42, set())
        self.assertEqual(self.invoke(tools, candidate="candidate-2")["error"]["code"], "OUTSIDE_ROUND")
        self.assertEqual(self.invoke(tools, seed=43)["error"]["code"], "OUTSIDE_ROUND")
        self.assertEqual(self.invoke(tools, seed=True)["error"]["code"], "INVALID_ARGUMENTS")
        self.assertEqual(tools.invoke("run_experiment", {"candidate_id": "candidate-1"})["error"]["code"],
                         "INVALID_ARGUMENTS")
        self.assertEqual(self.submitted, [])

    def test_duplicate_source_is_rejected_without_consuming_an_attempt(self):
        tools = RoundResearch(self.research, "candidate-1", 42, {self.baseline["source_hash"]})
        result = self.invoke(tools)
        self.assertEqual(result["error"]["code"], "DUPLICATE_CANDIDATE")
        self.assertEqual(self.research.usage["experiment_attempts"], 0)
        self.assertEqual(self.submitted, [])

    def test_successful_attempt_cannot_be_repeated_even_through_the_cache(self):
        self.patch_model()
        tools = RoundResearch(self.research, "candidate-1", 42, {self.baseline["source_hash"]})
        first = self.invoke(tools)
        self.assertTrue(first["ok"])
        self.assertEqual(self.invoke(tools)["error"]["code"], "ROUND_EXPERIMENT_LIMIT")
        self.assertEqual(len(self.submitted), 1)
        self.assertEqual(self.research.usage["experiment_attempts"], 1)

    def test_failed_attempt_still_consumes_the_one_round_slot(self):
        self.fail_runner = True
        self.patch_model()
        tools = RoundResearch(self.research, "candidate-1", 42, set())
        first = self.invoke(tools)
        self.assertEqual(first["result"]["status"], "failed")
        self.assertEqual(self.invoke(tools)["error"]["code"], "ROUND_EXPERIMENT_LIMIT")
        self.assertEqual(len(self.submitted), 1)

    def test_next_round_uses_the_same_strategy_budget(self):
        self.patch_model(2)
        one = RoundResearch(self.research, "candidate-1", 42, set())
        self.assertTrue(self.invoke(one)["ok"])
        self.patch_model(3)
        two = RoundResearch(self.research, "candidate-2", 42, set())
        self.assertTrue(self.invoke(two, "candidate-2")["ok"])
        self.assertEqual(two.usage["experiment_attempts"], 2)
        self.assertEqual(len(two.experiments), 2)

    def test_restoring_best_snapshot_does_not_charge_an_agent_patch(self):
        self.patch_model(2)
        patches = self.research.usage["patch_attempts"]
        restored = restore_snapshot(self.research, self.baseline["source_hash"])
        self.assertEqual(restored["source_hash"], self.baseline["source_hash"])
        self.assertEqual(self.workspace.read("src/model.py")["text"], self.files["src/model.py"])
        self.assertEqual(self.research.usage["patch_attempts"], patches)

    def test_tampered_snapshot_is_rejected_before_editable_files_are_written(self):
        self.patch_model(2)
        current = self.workspace.read("src/model.py")["text"]
        path = Path(self.baseline["directory"]) / "configs/default.json"
        path.write_text('{"epochs":99}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot source hash"):
            restore_snapshot(self.research, self.baseline["source_hash"])
        self.assertEqual(self.workspace.read("src/model.py")["text"], current)

    def test_readonly_change_is_not_silently_repaired(self):
        self.patch_model(2)
        current = self.workspace.read("src/model.py")["text"]
        readonly = self.workspace.root / "src/dependency.py"
        readonly.write_text("constant = 99\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "read-only"):
            restore_snapshot(self.research, self.baseline["source_hash"])
        self.assertEqual(self.workspace.read("src/model.py")["text"], current)
        self.assertEqual(readonly.read_text(encoding="utf-8"), "constant = 99\n")

    def test_restore_normalizes_newlines_and_rejects_traversal(self):
        snapshot_model = Path(self.baseline["directory"]) / "src/model.py"
        snapshot_model.write_bytes(self.files["src/model.py"].replace("\n", "\r\n").encode())
        self.patch_model(2)
        restored = restore_snapshot(self.research, self.baseline["source_hash"])
        self.assertEqual(restored["source_hash"], self.baseline["source_hash"])
        with self.assertRaises(ValueError):
            restore_snapshot(self.research, "../outside")


if __name__ == "__main__":
    unittest.main()
