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
