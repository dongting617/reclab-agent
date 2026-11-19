"""Trusted scorer dispatch contracts; no model code, worker or test-set access."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reclab_agent.contracts import TaskSpec
from reclab_agent.datasets import create_synthetic_bundle
from reclab_agent.research.configuration import TaskConfiguration
from reclab_agent.research.inspection import EvaluatorDescription, TaskAdapterRegistry, inspect_task


class RegistryScoringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="score-", dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = create_synthetic_bundle(self.root / "dataset")
        self.predictions = {}
        catalog = self.bundle.manifest["catalog"]
        for user, row in self.bundle.valid.items():
            eligible = [item for item in catalog if item not in row["history"] and item != row["target"]]
            self.predictions[user] = [row["target"], *eligible[:19]]
        self.registry = TaskAdapterRegistry()

    def description(self, bundle):
        return EvaluatorDescription("a" * 64, ("quality",), {"split": "valid"})

    def custom_report(self):
        return {"protocol_hash": "a" * 64, "evaluation_hash": "b" * 64,
                "metrics": {"users": len(self.predictions), "quality": 0.5}}

    def register_result(self, report):
        registry = TaskAdapterRegistry(include_defaults=False)
        registry.register_evaluator("custom", self.description, score=lambda dataset, predictions: report)
        return registry

    def test_builtin_recomputes_validation_score_and_never_requests_test(self):
        private_test = self.bundle.root / "private/test.json"
        private_test.write_text("private test labels must not be read", encoding="utf-8")
        expected = self.bundle.grade(self.predictions, "valid")
        with patch.object(self.bundle, "grade", wraps=self.bundle.grade) as grade:
            report = self.registry.score_predictions("next_item_ranking_v1", self.bundle, self.predictions)
        grade.assert_called_once_with(self.predictions, "valid")
        self.assertEqual(report, expected)
        self.assertEqual(report["metrics"]["ndcg@20"], 1.0)
        self.assertEqual(report["metrics"]["recall@20"], 1.0)
        self.assertTrue(self.registry.can_score("next_item_ranking_v1"))

    def test_builtin_rejects_bad_user_coverage_and_ineligible_rankings(self):
        user = next(iter(self.predictions))
        cases = []
        missing = copy.deepcopy(self.predictions)
        del missing[user]
        cases.append(missing)
        extra = copy.deepcopy(self.predictions)
        extra["unknown-user"] = list(extra[user])
        cases.append(extra)
        for first in (self.bundle.valid[user]["history"][0], 0, 41, True):
            invalid = copy.deepcopy(self.predictions)
            invalid[user][0] = first
            cases.append(invalid)
        duplicate = copy.deepcopy(self.predictions)
        duplicate[user][1] = duplicate[user][0]
        cases.append(duplicate)
        short = copy.deepcopy(self.predictions)
        short[user] = short[user][:1]
        cases.append(short)
        for predictions in cases:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                self.registry.score_predictions("next_item_ranking_v1", self.bundle, predictions)

    def test_unknown_or_description_only_evaluators_cannot_score(self):
        self.assertFalse(self.registry.can_score("unknown.module"))
        with self.assertRaisesRegex(ValueError, "not registered"):
            self.registry.score_predictions("unknown.module", self.bundle, self.predictions)
        self.registry.register_evaluator("description_only", self.description)
        self.assertFalse(self.registry.can_score("description_only"))
        with self.assertRaisesRegex(ValueError, "no scoring callback"):
            self.registry.score_predictions("description_only", self.bundle, self.predictions)

    def test_custom_trusted_scorer_receives_inputs_and_returns_bound_report(self):
        calls, expected = [], self.custom_report()
        expected["status"] = "completed"

        def score(bundle, predictions):
            calls.append((bundle, predictions))
            return expected

        self.registry.register_evaluator("custom", self.description, score=score)
        report = self.registry.score_predictions("custom", self.bundle, self.predictions)
        self.assertEqual(calls, [(self.bundle, self.predictions)])
        self.assertEqual(report, expected)

    def test_report_requires_registered_protocol_and_evaluation_context_hashes(self):
        cases = []
        for key, value in (("protocol_hash", "c" * 64), ("evaluation_hash", "B" * 64),
                           ("evaluation_hash", "short")):
            report = self.custom_report()
            report[key] = value
            cases.append(report)
        missing = self.custom_report()
        del missing["evaluation_hash"]
        cases.append(missing)
        for report in cases:
            with self.subTest(report=report), self.assertRaises(ValueError):
                self.register_result(report).score_predictions("custom", self.bundle, self.predictions)

    def test_report_requires_complete_finite_metrics_and_coherent_user_count(self):
        cases = []
        for key, value in (("quality", float("nan")), ("quality", float("inf")),
                           ("quality", True), ("quality", 10**400),
                           ("users", 0), ("users", True),
                           ("users", len(self.predictions) - 1), ("users", 12.0)):
            report = self.custom_report()
            report["metrics"][key] = value
            cases.append(report)
        missing = self.custom_report()
        del missing["metrics"]["quality"]
        cases.append(missing)
        extra_nonfinite = self.custom_report()
        extra_nonfinite["metrics"]["diagnostic"] = float("inf")
        cases.append(extra_nonfinite)
        for report in cases:
            with self.subTest(report=report), self.assertRaises(ValueError):
                self.register_result(report).score_predictions("custom", self.bundle, self.predictions)

    def test_failed_or_non_json_scoring_cannot_be_a_completed_report(self):
        cases = [[], None]
        for key, value in (("status", "failed"), ("status", "running"), ("ok", False),
                           ("error", "scoring failed"), ("failure", "no scores"),
                           ("metadata", object())):
            report = self.custom_report()
            report[key] = value
            cases.append(report)
        for report in cases:
            with self.subTest(report=report), self.assertRaises(ValueError):
                self.register_result(report).score_predictions("custom", self.bundle, self.predictions)

    def test_prediction_users_must_be_a_nonempty_plain_mapping(self):
        expected = self.custom_report()
        registry = self.register_result(expected)
        for predictions in ({}, [], {1: [1]}, {" ": [1]}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                registry.score_predictions("custom", self.bundle, predictions)

    def test_bad_or_duplicate_registration_does_not_replace_existing_callbacks(self):
        registry = self.register_result(self.custom_report())
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register_evaluator("custom", self.description, score=lambda dataset, predictions: {})
        self.assertEqual(registry.score_predictions("custom", self.bundle, self.predictions), self.custom_report())
        with self.assertRaises(TypeError):
            registry.register_evaluator("invalid", self.description, score=object())
        self.assertFalse(registry.can_score("invalid"))
        with self.assertRaisesRegex(ValueError, "not registered"):
            registry.describe_evaluator("invalid", self.bundle)

    def configuration(self):
        source = self.root / "source"
        (source / "src").mkdir(parents=True)
        (source / "configs").mkdir()
        (source / "src/model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
        for filename in ("train.py", "predict.py"):
            (source / "src" / filename).write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
        config = {"model": {"file": "src/model.py", "class": "Model", "kwargs": {}}, "max_length": 3}
        (source / "configs/default.json").write_text(json.dumps(config), encoding="utf-8")
        return TaskConfiguration(
            configuration_path=self.root / "task.json",
            task=TaskSpec("scoring-task", "Optimize validation ranking", self.bundle.dataset_id, "baseline"),
            source_directory=source, config_name="default",
            editable_files=("src/model.py", "configs/default.json"),
            dataset_directory=self.bundle.root, dataset_adapter="next_item_v1",
            expected_data_hash=self.bundle.data_hash, evaluator_id="next_item_ranking_v1",
            primary_metric="ndcg@20", guardrail_metrics=("recall@20",),
            candidate_ids=("candidate-1",), seeds=(42,), model_env="RECLAB_SCORE_MODEL",
            api_key_env="RECLAB_SCORE_KEY", execution_backend="docker", execution_policy_path=None,
            output_directory=self.root / "results",
        )

    def test_inspection_distinguishes_description_registration_from_actual_scoring(self):
        config = self.configuration()
        builtin = inspect_task(config, registry=self.registry)
        self.assertTrue(builtin["inputs_valid"])
        self.assertTrue(builtin["evaluation"]["scoring_configured"])
        self.assertNotIn("evaluator_scoring_not_configured", builtin["pending"])
        self.registry.register_evaluator("description_only", self.description)
        description_only = inspect_task(replace(config, evaluator_id="description_only",
                                               primary_metric="quality", guardrail_metrics=()),
                                        registry=self.registry)
        self.assertTrue(description_only["inputs_valid"])
        self.assertFalse(description_only["ready_to_run"])
        self.assertFalse(description_only["evaluation"]["scoring_configured"])
        self.assertIn("evaluator_scoring_not_configured", description_only["pending"])
        self.assertFalse(config.output_directory.exists())


if __name__ == "__main__":
    unittest.main()
