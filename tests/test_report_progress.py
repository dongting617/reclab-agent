"""Progress markers and report conclusions share verified acceptance rules."""
from copy import deepcopy
import unittest
from reclab_agent.reporting import build_report_view
from tests import test_user_report as fixtures


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UserReportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def view(self, report):
        return build_report_view(report, output_directory=self.fixture.root, context=self.fixture.context)

    def test_milestones_keep_order_declines_ties_failures_and_previous_best(self):
        f = self.fixture
        low = f.entry("low", .07, .16)
        first = f.entry("first", .09, .16)
        tie = f.entry("tie", .09, .16)
        below_best = f.entry("below", .085, .16)
        fail = f.entry("failed", status="failed")
        second = f.entry("second", .11, .17)
        report = f.report(low, first, tie, below_best, fail, second)
        original = deepcopy(report)
        view = self.view(report)
        points = view["progress"]["groups"][0]["points"]
        self.assertEqual([p["id"] for p in points], ["low", "first", "tie", "below", "failed", "second"])
        self.assertEqual([p["id"] for p in points if p["milestone"]], ["first", "second"])
        self.assertIsNone(points[4]["value"])
        self.assertAlmostEqual(points[4]["best"], .09)
        self.assertAlmostEqual(points[-1]["previous_delta"], .02)
        self.assertEqual(view["best"]["id"], "second")
        self.assertIn("second", view["summary"])
        self.assertEqual(report, original)

    def test_missing_scope_unverified_scores_and_guardrail_never_win(self):
        f = self.fixture
        for key in ("seed", "dataset_id", "protocol_hash", "image"):
            candidate = deepcopy(f.candidate)
            candidate["record"].pop(key)
            view = self.view(f.report(candidate))
            self.assertFalse(view["improved"], key)
            self.assertIsNone(view["progress"]["groups"][0]["points"][0]["value"], key)
        view = build_report_view(f.report(), context=f.context)
        self.assertFalse(view["improved"])
        self.assertIsNone(view["progress"]["groups"][0]["baseline"]["value"])
        unknown_environment = deepcopy(f.report())
        for row in unknown_environment["experiments"]:
            row["record"].pop("image", None)
        self.assertFalse(self.view(unknown_environment)["improved"])
        candidate = f.entry("unsafe", .2, .01)
        view = self.view(f.report(candidate))
        point = view["progress"]["groups"][0]["points"][0]
        self.assertEqual(point["value"], .2)
        self.assertFalse(point["qualified"])
        self.assertFalse(view["improved"])

    def test_required_process_and_audit_are_not_inferred_from_high_score(self):
        f = self.fixture
        report = f.report()
        report["hypotheses"] = [{"candidate_id": "candidate-safe", "process_metrics": [
            {"name": "epochs", "purpose": "required"}]}]
        view = self.view(report)
        self.assertFalse(view["improved"])
        candidate = report["experiments"][1]
        candidate["eligible"] = True
        candidate["process_metrics"] = {"passes_required": True, "definition_hash": "fixture",
            "metrics": [{"name": "epochs", "purpose": "required", "passes": True, "status": "observed"}]}
        self.assertTrue(self.view(report)["improved"])
        report["context"] = {"agent_mode": "research_audit"}
        context = {**f.context, "agent_mode": "research_audit"}
        for audit in (None, {"status": "completed", "audit_approved": True, "audit": {"verdict": "rejected"}}):
            report["audits"] = {"candidate-safe": audit}
            view = build_report_view(report, output_directory=f.root, context=context)
            self.assertFalse(view["improved"])
        report["audits"] = {"candidate-safe": {"status": "completed", "audit_approved": True,
            "audit": {"verdict": "approved", "summary": "fixture"}}}
        view = build_report_view(report, output_directory=f.root, context=context)
        self.assertTrue(view["improved"])

    def test_zero_baseline_and_unconfirmed_stop(self):
        f = self.fixture
        baseline = f.entry("zero", 0, .15)
        report = f.report()
        report["baseline_id"] = "zero"
        report["experiments"][0] = baseline
        view = self.view(report)
        point = view["progress"]["groups"][0]["points"][0]
        self.assertTrue(point["milestone"])
        self.assertIsNone(point["relative"])
        report["requires_reconciliation"] = True
        self.assertFalse(self.view(report)["improved"])


if __name__ == "__main__":
    unittest.main()
