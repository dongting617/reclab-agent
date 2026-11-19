"""Review, evidence and experiment contracts; no model calls or job execution."""
from dataclasses import replace
import json
import tempfile
import unittest

from reclab_agent.contracts import Budget, TaskSpec
from reclab_agent.research import (
    EvidenceRef, ExperimentRecord, PlanReview, ResearchClaim, ResearchConclusion,
    ResearchPlan, ResearchStage, require_plan_approval, stage_tool_names,
)
from reclab_agent.storage import ArtifactStore
from reclab_agent.tools import tool_catalog


class ResearchContractTests(unittest.TestCase):
    def setUp(self):
        self.evidence = EvidenceRef("evidence/baseline.json", "a" * 64,
                                    "/metrics/ndcg@20", "Beauty valid; same users and protocol")
        self.plan = ResearchPlan(
            plan_id="research-1", task_id="task-1", hypothesis="Reweighting may improve rare-item recall",
            dataset_id="beauty", data_profile="beauty-small", baseline_id="b1", protocol_hash="b" * 64,
            primary_metric="ndcg@20", editable_files=("src/model.py", "configs/default.json"),
            comparison="Baseline and candidate use the same data, seed and training budget",
            expected_observation="Rare-item recall improves without violating the primary metric guardrail",
            rejection_criteria="Reject if the primary metric violates the predeclared guardrail",
            stop_conditions=("training budget exhausted", "repeated error without new evidence"),
            evidence=(self.evidence,), guardrail_metrics=("recall@20",),
        )

    def review(self, plan=None):
        return PlanReview((plan or self.plan).plan_hash, "approved", "user", "2026-11-18T12:00:00+08:00")

    def experiment(self, **overrides):
        values = dict(
            experiment_id="exp-1", plan_hash=self.plan.plan_hash, candidate_id="candidate-1",
            source_hash="c" * 64, config_hash="d" * 64, dataset_id="beauty", protocol_hash="b" * 64,
            image="example/reclab@sha256:" + "e" * 64, seed=42,
        )
        return ExperimentRecord(**{**values, **overrides})

    def test_plan_roundtrip_can_be_stored_without_changing_existing_storage(self):
        restored = ResearchPlan.from_dict(json.loads(json.dumps(self.plan.to_dict())))
        self.assertEqual(restored, self.plan)
        self.assertEqual(restored.plan_hash, self.plan.plan_hash)
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(directory)
            artifact = store.put_json("research-plan", self.plan.to_dict())
            self.assertEqual(ResearchPlan.from_dict(store.read_json(artifact)), self.plan)

    def test_review_is_required_and_bound_to_complete_plan(self):
        approved = self.review()
        require_plan_approval(self.plan, approved)
        for review in (None, replace(approved, decision="rejected")):
            with self.subTest(review=review), self.assertRaises(ValueError):
                require_plan_approval(self.plan, review)
        for changed in (
            replace(self.plan, hypothesis="A different mechanism"),
            replace(self.plan, editable_files=("src/other.py",)),
            replace(self.plan, seeds=(43,)),
            replace(self.plan, candidate_ids=("candidate-2",)),
            replace(self.plan, budget=Budget(max_training_jobs=4)),
            replace(self.plan, revision=2),
        ):
            with self.subTest(plan=changed.plan_hash), self.assertRaises(ValueError):
                require_plan_approval(changed, approved)

    def test_repairs_remain_in_scope_and_confirmation_requires_explicit_scope(self):
        self.plan.check_scope("candidate-1", ("src/model.py",), 42)
        self.plan.check_scope("candidate-1", ("src/model.py", "configs/default.json"), 42)
        for candidate, paths, seed in (
            ("candidate-2", ("src/model.py",), 42),
            ("candidate-1", ("src/scorer.py",), 42),
            ("candidate-1", ("src/model.py",), 43),
            ("candidate-1", ("src/model.py",), True),
        ):
            with self.subTest(candidate=candidate, paths=paths, seed=seed), self.assertRaises(ValueError):
                self.plan.check_scope(candidate, paths, seed)
        with self.assertRaises(ValueError):
            self.plan.check_scope("candidate-1", (), 42, confirmation=True)
        replace(self.plan, confirmation_included=True).check_scope("candidate-1", (), 42, confirmation=True)

    def test_plan_cannot_expand_task_budget_or_change_registered_task(self):
        task = TaskSpec("task-1", "improve rare items", "beauty", "b1")
        self.plan.validate_for_task(task)
        for plan in (replace(self.plan, budget=Budget(max_training_jobs=4)),
                     replace(self.plan, baseline_id="b2")):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                plan.validate_for_task(task)

    def test_hypothesis_first_is_legal_but_observations_need_references(self):
        plan = replace(self.plan, evidence=())
        require_plan_approval(plan, self.review(plan))
        ResearchClaim("This might improve coverage", "hypothesis")
        ResearchClaim("A plausible explanation", "inference", (self.evidence,))
        with self.assertRaises(ValueError):
            ResearchClaim("The metric improved", "observation")

    def test_program_failure_does_not_become_a_scored_negative_result(self):
        failed = self.experiment(status="failed", program_validation="failed", failure="shape mismatch")
        self.assertIsNone(failed.score)
        with self.assertRaises(ValueError):
            replace(failed, score=self.evidence)
        with self.assertRaises(ValueError):
            self.experiment(status="completed", program_validation="passed")
        for status in ("planned", "running", "failed", "cancelled", "timed_out", "lost"):
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.experiment(status=status, program_validation="passed", score=self.evidence)
        with self.assertRaises(ValueError):
            self.experiment(status="completed", program_validation="passed", training_job_id="train-1",
                            prediction_job_id="predict-1", score=self.evidence, failure="training crashed")

    def test_json_arrays_cannot_be_replaced_by_strings_or_objects(self):
        for field in ("editable_files", "stop_conditions", "guardrail_metrics", "candidate_ids", "seeds", "evidence"):
            for value in ("budget", {"budget": 1}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ResearchPlan.from_dict({**self.plan.to_dict(), field: value})
        conclusion = ResearchConclusion("task-1", self.plan.plan_hash, "inconclusive", (), "budget exhausted")
        for field in ("findings", "experiment_ids", "limitations", "next_steps"):
            for value in ("budget", {"budget": 1}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ResearchConclusion.from_dict({**conclusion.to_dict(), field: value})

    def test_valid_negative_experiment_and_inconclusive_conclusion_are_normal_results(self):
        completed = self.experiment(status="completed", program_validation="passed",
                                    training_job_id="train-1", prediction_job_id="predict-1", score=self.evidence)
        self.assertEqual(ExperimentRecord.from_dict(json.loads(json.dumps(completed.to_dict()))), completed)
        claim = ResearchClaim("Validation NDCG decreased under the registered comparison", "observation", (self.evidence,))
        conclusion = ResearchConclusion("task-1", self.plan.plan_hash, "reject", (claim,),
                                        "planned comparison completed", (completed.experiment_id,),
                                        ("One development seed; no stability claim",))
        self.assertEqual(ResearchConclusion.from_dict(json.loads(json.dumps(conclusion.to_dict()))), conclusion)
        uncertain = replace(conclusion, decision="inconclusive", stop_reason="budget exhausted")
        self.assertFalse(uncertain.confirmed)
        with self.assertRaises(ValueError):
            replace(uncertain, findings=(), confirmed=True)

    def test_plan_rejects_traversal_and_non_candidate_file_paths(self):
        for path in ("../src/model.py", "src/../grader.py", "src//model.py", "C:/src/model.py",
                     "src\\model.py", "grader.py", "src/model.pkl"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(self.plan, editable_files=(path,))
        with self.assertRaises(ValueError):
            replace(self.plan, seeds=(True,))

    def test_stage_contract_keeps_mutation_and_submission_out_of_analysis(self):
        self.assertEqual(len(ResearchStage), 8)
        catalog_names = {capability.name for capability in tool_catalog()}
        for stage in ResearchStage:
            self.assertLessEqual(stage_tool_names(stage), catalog_names)
        for stage in (ResearchStage.CONTEXT, ResearchStage.INVESTIGATE, ResearchStage.REVIEW,
                      ResearchStage.ANALYZE, ResearchStage.REPORT):
            self.assertFalse({"apply_patch", "submit_experiment"} & stage_tool_names(stage))
        self.assertIn("apply_patch", stage_tool_names("implement"))
        self.assertIn("submit_experiment", stage_tool_names("experiment"))
        self.assertIn("submit_experiment", stage_tool_names("confirm"))
        self.assertNotIn("apply_patch", stage_tool_names("confirm"))
