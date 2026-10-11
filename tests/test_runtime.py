import json
from pathlib import Path
import shutil
import unittest

from reclab_agent import AgentRuntime, ModelTurn, RecLabToolbox
from reclab_agent.evaluation import evaluate_run, evaluate_tool_selection
from reclab_agent.runtime import function_tools


ROOT = Path.cwd()
TEMP = ROOT / ".test-tmp"


class ScriptedBackend:
    def __init__(self, turns, failures=0):
        self.turns = list(turns)
        self.failures = failures
        self.calls = 0

    def respond(self, **_kwargs):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise TimeoutError("injected timeout")
        return self.turns.pop(0)


def tool_turn(name, arguments, call_id="call-1"):
    return ModelTurn(output=[{
        "type": "function_call", "name": name,
        "arguments": json.dumps(arguments), "call_id": call_id,
    }], output_text="", response_id=f"response-{call_id}")


class AgentRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEMP.mkdir(exist_ok=True)
        cls.toolbox = RecLabToolbox(ROOT)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEMP, ignore_errors=True)

    def test_strict_function_schemas_are_closed_and_required(self):
        for definition in function_tools(self.toolbox):
            schema = definition["parameters"]
            self.assertTrue(definition["strict"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_agent_runs_tool_loop_and_writes_evidence_trace(self):
        trace = TEMP / "beam-trace.jsonl"
        backend = ScriptedBackend([
            tool_turn("diagnose_beam_gap", {}, "beam-call"),
            ModelTurn(output=[{"type": "message"}],
                      output_text="Beam 不是主要瓶颈，证据见差距报告。", response_id="final"),
        ])
        result = AgentRuntime(self.toolbox, backend, trace_path=trace).run(
            "Beam Search 是主要瓶颈吗？", task_id="beam-case")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], 1)
        self.assertIn(self.toolbox.GAP_REPORT, result["evidence_paths"])
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([event["event"] for event in events],
                         ["run_started", "model_turn", "tool_call", "model_turn", "run_finished"])
        metrics = evaluate_run(result, events, ["diagnose_beam_gap"])
        self.assertEqual(metrics["tool_selection"]["f1"], 1.0)
        self.assertEqual(metrics["tool_error_rate"], 0.0)

    def test_model_timeout_is_retried(self):
        backend = ScriptedBackend([
            ModelTurn(output=[{"type": "message"}], output_text="done")
        ], failures=1)
        result = AgentRuntime(self.toolbox, backend, model_retries=1).run("检查证据")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(backend.calls, 2)

    def test_invalid_arguments_are_returned_to_model(self):
        malformed = ModelTurn(output=[{
            "type": "function_call", "name": "diagnose_beam_gap",
            "arguments": "{", "call_id": "bad-call",
        }], output_text="")
        backend = ScriptedBackend([
            malformed,
            ModelTurn(output=[{"type": "message"}], output_text="参数错误，已安全停止。"),
        ])
        result = AgentRuntime(self.toolbox, backend).run("执行诊断")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], 1)

    def test_step_limit_stops_repeated_calls(self):
        backend = ScriptedBackend([
            tool_turn("diagnose_beam_gap", {}, "one"),
            tool_turn("diagnose_beam_gap", {}, "two"),
        ])
        result = AgentRuntime(self.toolbox, backend, max_steps=2).run("不要停止")
        self.assertEqual(result["status"], "step_limit_exceeded")
        self.assertEqual(result["tool_calls"], 2)

    def test_tool_selection_metric_penalizes_extra_calls(self):
        metric = evaluate_tool_selection(
            ["diagnose_beam_gap", "list_evidence"], ["diagnose_beam_gap"])
        self.assertEqual(metric["precision"], 0.5)
        self.assertEqual(metric["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
