import copy
import math
import unittest

from reclab_agent.grading import RankingProtocol, compare_reports, grade_rankings


class RankingGradingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = RankingProtocol("tiny", "valid", (1, 2, 3, 4), (1, 2))
        self.targets = {"a": 2, "b": 4}
        self.histories = {"a": [1], "b": [1]}
        self.predictions = {"a": [2, 3], "b": [3, 4]}

    def grade(self, predictions=None, targets=None, histories=None, protocol=None):
        return grade_rankings(self.predictions if predictions is None else predictions,
                              self.targets if targets is None else targets,
                              self.histories if histories is None else histories,
                              self.protocol if protocol is None else protocol)

    def test_known_values_and_missing_target_is_zero(self):
        report = self.grade()
        self.assertEqual(report["metrics"]["users"], 2)
        self.assertEqual(report["metrics"]["recall@1"], 0.5)
        self.assertEqual(report["metrics"]["recall@2"], 1.0)
        self.assertAlmostEqual(report["metrics"]["ndcg@2"], (1 + 1 / math.log2(3)) / 2)
        self.assertEqual(report["per_user"]["b"]["rank"], 2)
        miss = self.grade({"a": [3, 4], "b": [2, 3]})
        self.assertIsNone(miss["per_user"]["a"]["rank"])
        self.assertEqual(miss["metrics"]["ndcg@2"], 0)

    def test_order_does_not_change_protocol_or_report(self):
        reversed_protocol = RankingProtocol("tiny", "valid", (4, 3, 2, 1), (2, 1))
        reversed_report = self.grade(dict(reversed(list(self.predictions.items()))),
                                     dict(reversed(list(self.targets.items()))),
                                     {"b": [1, 1], "a": [1]}, reversed_protocol)
        self.assertEqual(self.grade(), reversed_report)

    def test_small_catalog_requires_all_eligible_items(self):
        protocol = RankingProtocol("tiny", "valid", (1, 2, 3))
        report = grade_rankings({"a": [3, 2]}, {"a": 2}, {"a": [1]}, protocol)
        self.assertEqual(report["metrics"]["recall@20"], 1)
        with self.assertRaises(ValueError):
            grade_rankings({"a": [2]}, {"a": 2}, {"a": [1]}, protocol)

    def test_user_coverage_and_ids_are_strict(self):
        for predictions in ({"a": [2, 3]}, {**self.predictions, "c": [2, 3]},
                            {1: [2, 3], "b": [3, 4]}, {"": [2, 3], "b": [3, 4]}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                self.grade(predictions)
        with self.assertRaises(ValueError):
            self.grade(histories={"a": [1]})
        with self.assertRaises(ValueError):
            grade_rankings({}, {}, {}, self.protocol)

    def test_duplicate_invalid_short_and_masked_predictions_rejected(self):
        for values in ([2, 2], [2, 99], [2, 1], [2], [True, 3], [float("nan"), 3], (2, 3)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.grade({"a": values, "b": [3, 4]})

    def test_target_and_history_integrity(self):
        for target in (True, float("nan"), 99, 1):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.grade(targets={"a": target, "b": 4})
        for history in ([99], [True], [float("nan")]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                self.grade(histories={"a": history, "b": [1]})
        protocol = RankingProtocol("tiny", "valid", (1, 2), (1,), mask_seen=False)
        report = grade_rankings({"a": [1]}, {"a": 1}, {"a": [1]}, protocol)
        self.assertEqual(report["metrics"]["recall@1"], 1)

    def test_protocol_rejects_noncanonical_types(self):
        for kwargs in ({"catalog": (1, True)}, {"catalog": (1, 1)},
                       {"catalog": (1, float("nan"))}, {"ks": (0,)},
                       {"ks": (True,)}, {"ks": ()}, {"ks": (1, 1)},
                       {"mask_seen": 1}, {"split": " "}, {"catalog": [1, 2]}):
            values = {"dataset_id": "tiny", "split": "valid", "catalog": (1, 2), **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RankingProtocol(**values)

    def test_paired_comparison_aligns_by_id_and_is_reproducible(self):
        baseline = self.grade()
        candidate = self.grade({"b": [4, 3], "a": [2, 3]})
        first = compare_reports(baseline, candidate, bootstrap_samples=200, seed=42)
        self.assertEqual(first, compare_reports(baseline, candidate, 200, 42))
        self.assertEqual(first["metrics"]["recall@1"]["mean_delta"], 0.5)
        self.assertEqual(first["metrics"]["recall@1"]["ci95"], [0, 1])
        self.assertEqual(first["metrics"]["recall@2"]["ci95"], [0, 0])
        negative = compare_reports(candidate, baseline, 200, 42)
        self.assertEqual(negative["metrics"]["recall@1"]["mean_delta"], -0.5)
        self.assertEqual(negative["metrics"]["recall@1"]["ci95"], [-1, 0])

    def test_different_protocol_users_and_targets_cannot_be_compared(self):
        baseline = self.grade()
        different_split = self.grade(protocol=RankingProtocol("tiny", "test", (1, 2, 3, 4), (1, 2)))
        different_k = self.grade(protocol=RankingProtocol("tiny", "valid", (1, 2, 3, 4), (2,)))
        different_target = self.grade(targets={"a": 3, "b": 4})
        different_users = grade_rankings({"x": [2, 3]}, {"x": 2}, {"x": [1]}, self.protocol)
        for candidate in (different_split, different_k, different_target, different_users):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                compare_reports(baseline, candidate)

    def test_comparison_rejects_nonfinite_or_corrupt_metrics(self):
        baseline = self.grade()
        for value in (float("nan"), float("inf"), True, 5):
            broken = copy.deepcopy(baseline)
            broken["per_user"]["a"]["recall@1"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_reports(baseline, broken)
        broken = copy.deepcopy(baseline)
        broken["metrics"]["recall@1"] = 0.25
        with self.assertRaises(ValueError):
            compare_reports(baseline, broken)
        broken = copy.deepcopy(baseline)
        broken["protocol"]["split"] = "test"
        with self.assertRaises(ValueError):
            compare_reports(baseline, broken)
        for kwargs in ({"bootstrap_samples": 0}, {"bootstrap_samples": True}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compare_reports(baseline, baseline, **kwargs)


if __name__ == "__main__":
    unittest.main()
