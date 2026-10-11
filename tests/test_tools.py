import json
from pathlib import Path
import unittest

from reclab_agent import RecLabToolbox, load_tasks, run_reference_suite
from reclab_agent.tools import json_pointer


ROOT = Path.cwd()


class RecLabToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = RecLabToolbox(ROOT)

    def test_path_traversal_is_blocked(self):
        result = self.tools.invoke("inspect_artifact", {"path": "../resume.pdf"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "PATH_OUTSIDE_ROOT")

    def test_json_pointer_escaping(self):
        document = {"values": {"/overall/ndcg@20": 0.1}}
        self.assertEqual(json_pointer(document, "/values/~1overall~1ndcg@20"), 0.1)

    def test_beam_diagnostic_uses_committed_evidence(self):
        result = self.tools.invoke("diagnose_beam_gap")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["search_suboptimal_misses"], 5)
        self.assertEqual(result["result"]["conclusion_code"], "SCORING_NOT_SEARCH")

    def test_frequency_diagnostic_identifies_low_frequency_gap(self):
        result = self.tools.invoke("diagnose_frequency_gap")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["conclusion_code"], "LOW_FREQUENCY_GAP_CONFIRMED")
        self.assertEqual(result["result"]["largest_b1_advantage_bucket"], "5-19")

    def test_development_reference_suite(self):
        tasks = load_tasks(ROOT / "tasks" / "dev.json")
        self.assertEqual(len(tasks), 15)
        report = run_reference_suite(self.tools, tasks)
        if report["passed"] != report["tasks"]:
            self.fail(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()

