import json
from pathlib import Path
import tempfile
import unittest
from reclab_agent.agents import ReActAgent
from reclab_agent.core import ModelTurn
from reclab_agent.tools import RecLabToolbox
from reclab_agent.evaluation.traces import evaluate_run, evaluate_tool_selection, replay_trace
from reclab_agent.core.session import function_tools


ROOT = Path(__file__).absolute().parents[1]


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


class ReActAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="reclab-react-")
        self.addCleanup(temporary.cleanup)
        self.temporary_root = Path(temporary.name)
        self.toolbox = RecLabToolbox(ROOT)

    def test_strict_function_schemas_are_closed_and_required(self):
        for definition in function_tools(self.toolbox):
            schema = definition["parameters"]
            self.assertTrue(definition["strict"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_agent_runs_tool_loop_and_writes_evidence_trace(self):
        trace = self.temporary_root / "beam-trace.jsonl"
        backend = ScriptedBackend([
            tool_turn("diagnose_beam_gap", {}, "beam-call"),
            ModelTurn(output=[{"type": "message"}],
                      output_text="Beam 不是主要瓶颈，证据见差距报告。", response_id="final"),
        ])
        result = ReActAgent(self.toolbox, backend, trace_path=trace).run(
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
        result = ReActAgent(self.toolbox, backend, model_retries=1).run("检查证据")
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
        result = ReActAgent(self.toolbox, backend).run("执行诊断")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["tool_calls"], 1)

    def test_step_limit_stops_repeated_calls(self):
        backend = ScriptedBackend([
            tool_turn("diagnose_beam_gap", {}, "one"),
            tool_turn("diagnose_beam_gap", {}, "two"),
        ])
        result = ReActAgent(self.toolbox, backend, max_steps=2).run("不要停止")
        self.assertEqual(result["status"], "step_limit_exceeded")
        self.assertEqual(result["tool_calls"], 2)

    def test_tool_selection_metric_penalizes_extra_calls(self):
        metric = evaluate_tool_selection(
            ["diagnose_beam_gap", "list_evidence"], ["diagnose_beam_gap"])
        self.assertEqual(metric["precision"], 0.5)
        self.assertEqual(metric["recall"], 1.0)

    def test_task_policy_only_exposes_allowed_tools(self):
        backend = ScriptedBackend([
            tool_turn("diagnose_frequency_gap", {}, "policy-call"),
            ModelTurn(output=[{"type": "message"}], output_text="已拒绝越权工具。"),
        ])
        trace = self.temporary_root / "policy-trace.jsonl"
        result = ReActAgent(self.toolbox, backend, trace_path=trace).run(
            "诊断 Beam Search", allowed_tools={"diagnose_beam_gap"})
        exposed = backend.calls
        self.assertEqual(exposed, 2)
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        tool_event = next(event for event in events if event["event"] == "tool_call")
        self.assertEqual(tool_event["result"]["error"]["code"], "TOOL_NOT_ALLOWED")
        metrics = evaluate_run(result, events, ["diagnose_beam_gap"])
        self.assertEqual(metrics["policy_violations"], 1)

    def test_saved_trace_can_be_replayed_offline(self):
        trace = self.temporary_root / "replay-trace.jsonl"
        evidence_path = self.toolbox.GAP_REPORT
        backend = ScriptedBackend([
            tool_turn("diagnose_beam_gap", {}, "beam-call"),
            ModelTurn(output=[{"type": "message"}],
                      output_text=f"结论来自 {evidence_path}。"),
        ])
        ReActAgent(self.toolbox, backend, trace_path=trace).run(
            "Beam Search 是主要瓶颈吗？", task_id="replay-case",
            allowed_tools={"diagnose_beam_gap"})
        report = replay_trace(trace, ["diagnose_beam_gap"])
        self.assertTrue(report["trace_valid"])
        self.assertEqual(report["metrics"]["tool_selection"]["f1"], 1.0)
        self.assertEqual(report["metrics"]["answer_evidence_citation_rate"], 1.0)
        self.assertEqual(report["metrics"]["policy_violations"], 0)


if __name__ == "__main__":
    unittest.main()
