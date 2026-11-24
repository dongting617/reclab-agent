"""Diagnostic contracts and trusted fixture extensions; no Torch/GPU required."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.experiments.diagnostics import evaluate_process_metrics, validate_definitions


class DiagnosticTests(unittest.TestCase):
    def test_training_summaries_reuse_logs_without_inventing_gradients(self):
        definitions = [{"name": "training_epochs_reached", "purpose": "required", "expected": "two epochs",
                        "operator": ">=", "threshold": 2},
                       {"name": "training_loss_final", "purpose": "diagnostic", "expected": "finite"},
                       {"name": "grad_nan_check", "purpose": "diagnostic", "expected": "observed"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "experiments/exp-1/train/epochs.jsonl"
            path.parent.mkdir(parents=True)
            raw = b'{"epoch":1,"train_loss":3.0}\n{"epoch":2,"train_loss":2.0}\n'
            path.write_bytes(raw)
            result = evaluate_process_metrics(root, {"experiment_id": "exp-1"}, definitions)
            self.assertTrue(result["passes_required"])
            self.assertEqual([row["value"] for row in result["metrics"]], [2, 2.0, None])
            self.assertEqual(result["metrics"][-1]["status"], "missing")
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(result["derived_evidence"][0]["collector"], "training_log_summary_v1")
            first_hash = result["definition_hash"]
            path.write_text('{"epoch":1,"train_loss":1.0}\n')
            second = evaluate_process_metrics(root, {"experiment_id": "exp-1"}, definitions)
            self.assertEqual(first_hash, second["definition_hash"])
            self.assertFalse(second["passes_required"])

    def test_training_summary_rejects_gaps_and_nonfinite_values(self):
        definitions = [{"name": "training_epochs_reached", "purpose": "diagnostic", "expected": "recorded"}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "experiments/exp-1/train/epochs.jsonl"
            path.parent.mkdir(parents=True)
            for text in ('{"epoch":2,"train_loss":1.0}\n', '{"epoch":1,"train_loss":NaN}\n'):
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, "contiguous epochs"):
                    evaluate_process_metrics(root, {"experiment_id": "exp-1"}, definitions)

    def test_worker_extension_definition_hash_and_numeric_output(self):
        source = Path(__file__).resolve().parents[1] / "backends/sasrec/src/runtime.py"
        spec = importlib.util.spec_from_file_location("diagnostic_worker_fixture", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=lambda: lambda function: function),
                                       "data": SimpleNamespace(canonical_json=canonical_json)}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "diagnostics").mkdir()
            path = root / "diagnostics/check.py"
            path.write_text("def evaluate(context):\n    return {'coverage': 0.75}\n", encoding="utf-8")
            module.__file__ = str(root / "runtime.py")
            context = {"selected": {"selection_valid_ndcg@20": .1}}
            result = module.collect_diagnostics(context)
            self.assertEqual(result["metrics"]["coverage"], .75)
            path.write_text("def evaluate(context):\n    return {'coverage': float('nan')}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.collect_diagnostics(context)

    def test_required_thresholds_are_checked_and_missing_evidence_fails(self):
        definitions = [{"name": "coverage", "purpose": "required", "expected": "at least half",
                        "operator": ">=", "threshold": .5}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = {"experiment_id": "exp-1"}
            self.assertFalse(evaluate_process_metrics(root, record, definitions)["passes_required"])
            path = root / "experiments/exp-1/predict/process_metrics.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"definition_hash": "a" * 64, "metrics": {"coverage": .75}}))
            self.assertTrue(evaluate_process_metrics(root, record, definitions)["passes_required"])
        with self.assertRaises(ValueError):
            validate_definitions([{**definitions[0], "threshold": float("inf")}])
