"""Frozen diagnostics, paragraph annotations, CPU adapters and honest costs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reclab_agent.core import ModelTurn
from reclab_agent.evaluation.costs import summarize_usage
from reclab_agent.evaluation.diagnostics import load_diagnostic_tasks, run_diagnostic_suite
from reclab_agent.evaluation.retrieval import evaluate_retrieval
from reclab_agent.integrations.text_models import TextModels

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_freeze_has_disjoint_scene_groups_and_reference_cases_pass(self):
        groups = []
        for split in ("dev", "frozen"):
            path = ROOT / "tasks" / ("diagnostic-"+split+".json")
            manifest, _ = load_diagnostic_tasks(path)
            self.assertEqual(len(manifest["tasks"]), 30)
            groups.append({x["scene_group"] for x in manifest["tasks"]})
            report = run_diagnostic_suite(path, self.root / (split+".json"))
            self.assertEqual((report["passed"], report["tasks"]), (30, 30))
            self.assertEqual(report["kind"], "reference_contract")
            self.assertEqual(report["model_requests"], 0)
        self.assertFalse(groups[0] & groups[1])

    def test_tampered_frozen_manifest_is_rejected(self):
        shutil.copytree(ROOT / "tasks", self.root / "tasks")
        path = self.root / "tasks/diagnostic-frozen.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["tasks"][0]["expected_decision"] = "retain"
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            load_diagnostic_tasks(path)

    def test_current_freeze_does_not_expose_scene_names_through_dataset_ids(self):
        for split in ("dev", "frozen"):
            path = ROOT / "tasks/diagnostic-v2" / ("diagnostic-" + split + ".json")
            manifest, _ = load_diagnostic_tasks(path)
            for row in manifest["tasks"]:
                for scope in row["evidence"]["scope"].values():
                    self.assertTrue(scope["dataset"].startswith("data-"))
                    self.assertNotIn(row["scene_group"], scope["dataset"])
            result = run_diagnostic_suite(path, self.root / ("anonymous-" + split + ".json"))
            self.assertEqual(result["passed"], 30)

    def test_labels_are_not_visible_to_model_or_toolbox_and_budget_persists(self):
        class ReadBackend:
            model = "unit-fixture"
            calls = 0
            def respond(inner, **kwargs):
                inner.calls += 1
                text = json.dumps(kwargs["input_items"])
                self.assertNotIn("expected_decision", text)
                self.assertNotIn("required_pointer", text)
                return ModelTurn(output=[{"type": "function_call", "name": "read_json_values",
                    "arguments": json.dumps({"path": "../diagnostic-frozen.json", "pointers": [""]}),
                    "call_id": "read"}], output_text="")
        backend = ReadBackend()
        output = self.root / "bounded.json"
        report = run_diagnostic_suite(ROOT / "tasks/diagnostic-dev.json", output, mode="react",
                                      backend=backend, max_requests=1, max_steps=2)
        self.assertEqual(backend.calls, 1)
        self.assertEqual(report["tasks"], 30)
        self.assertEqual(report["passed"], 0)
        self.assertEqual(report["status"], "partial")
        run_diagnostic_suite(ROOT / "tasks/diagnostic-dev.json", output, mode="react",
                             backend=backend, max_requests=1, max_steps=2)
        self.assertEqual(backend.calls, 1)

    def test_failed_review_cannot_pass_using_a_correct_executor_draft(self):
        from tests.test_agent_strategies import ScriptedBackend, text_turn
        backend = ScriptedBackend(text_turn('{"steps":["inspect evidence"]}'),
            ModelTurn(output=[{"type": "function_call", "name": "read_json_values", "call_id": "read",
                              "arguments": '{"path":"evidence/case.json","pointers":[""]}'}], output_text=""),
            text_turn('{"decision":"retain","reason":"observed metrics","citations":[{"path":"evidence/case.json","pointer":"/metrics"}]}'),
            text_turn('{"approved":"not-a-boolean","feedback":""}'))
        report = run_diagnostic_suite(ROOT / "tasks/diagnostic-dev.json", self.root / "review.json",
                                      mode="team", backend=backend, max_requests=4)
        first = report["results"][0]
        self.assertEqual(first["agent_status"], "invalid_review")
        self.assertTrue(first["decision_correct"])
        self.assertFalse(first["passed"])

    def test_repeated_provider_failure_stops_calls_but_keeps_all_tasks(self):
        NetworkError = type("APIConnectionError", (Exception,), {"__module__": "openai"})
        class BrokenBackend:
            model = "offline-error-fixture"
            calls = 0
            def respond(inner, **kwargs):
                inner.calls += 1
                raise NetworkError("unavailable")
        backend = BrokenBackend()
        report = run_diagnostic_suite(ROOT / "tasks/diagnostic-dev.json", self.root / "outage.json",
                                      mode="react", backend=backend)
        self.assertEqual(backend.calls, 3)
        self.assertEqual(report["tasks"], 30)
        self.assertEqual(sum(r["status"] == "not_run" for r in report["results"]), 27)
        self.assertEqual(report["status"], "partial")

    def test_bm25_paragraph_report_and_modified_corpus_rejection(self):
        report = evaluate_retrieval(ROOT / "tasks/retrieval.json")
        run = report["runs"][0]
        self.assertEqual(run["queries"], 13)
        self.assertGreater(run["recall@5"], .8)
        self.assertEqual(run["no_answer_empty_rate"], 1)
        self.assertTrue(all("sha256" in h for r in run["results"] for h in r["hits"]))
        shutil.copytree(ROOT / "tasks", self.root / "tasks")
        (self.root / "tasks/retrieval-corpus/baseline.md").write_text("tampered")
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.root / "tasks/retrieval.json")

    def test_missing_models_do_not_masquerade_as_dense_retrieval(self):
        with self.assertRaises(ValueError):
            evaluate_retrieval(ROOT / "tasks/retrieval.json", modes=("dense",))

    def test_explicit_prices_and_missing_usage_are_not_reported_as_free(self):
        events = [{"event": "model_turn", "usage": {"input_tokens": 1000, "output_tokens": 200,
                   "input_tokens_details": {"cached_tokens": 500}}},
                  {"event": "model_error"}, {"event": "model_turn", "usage": None}]
        prices = {"currency": "EXAMPLE", "input_per_million": 2, "output_per_million": 4,
                  "cached_input_per_million": 1}
        result = summarize_usage(events, prices)
        self.assertAlmostEqual(result["estimated_cost"], .0023)
        self.assertFalse(result["billing_verified"])
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["requests_without_usage"], 2)
        self.assertIsNone(summarize_usage(events)["estimated_cost"])
        with self.assertRaises(ValueError):
            summarize_usage(events, {**prices, "input_per_million": -1})

    def test_cpu_model_adapters_are_lazy_and_use_no_remote_code(self):
        class Array(list):
            def tolist(self):return list(self)
        class Encoder:
            def __init__(inner, name, **kwargs):
                inner.kwargs = kwargs
            def encode(inner, texts, **kwargs):return Array([[1., 0.] for x in texts])
        class Ranker(Encoder):
            def predict(inner, pairs, **kwargs):return Array([.5 for pair in pairs])
        module = SimpleNamespace(SentenceTransformer=Encoder, CrossEncoder=Ranker)
        with patch.dict("sys.modules", {"sentence_transformers": module}):
            models = TextModels()
            self.assertNotIn("encoder", models.__dict__)
            self.assertEqual(models.embed(["one"]), [[1., 0.]])
            self.assertEqual(models.rerank("query", ["doc"]), [.5])
            self.assertEqual(models.encoder.kwargs["device"], "cpu")
            self.assertFalse(models.ranker.kwargs["trust_remote_code"])
            self.assertTrue(models.encoder.kwargs["local_files_only"])

    def test_report_renders_cost_limits_in_plain_language(self):
        from reclab_agent.reporting import build_report_view, render_html, render_markdown
        usage = summarize_usage([{"event": "model_turn", "usage": None}])
        view = build_report_view({"status": "incomplete", "experiments": [], "resource_usage": usage})
        for rendered in (render_html(view), render_markdown(view)):
            self.assertIn("缺少 usage", rendered)
            self.assertIn("未核验服务商账单", rendered)
