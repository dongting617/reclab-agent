"""Strategy behavior with deterministic model replies and real tool dispatch."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from reclab_agent.agents import (
    PlanExecuteAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
    create_agent,
)
from reclab_agent.core.llm import ModelTurn
from reclab_agent.evaluation.traces import load_trace, replay_trace
from reclab_agent.tools import ToolSpec


def text_turn(text: str) -> ModelTurn:
    return ModelTurn(output=[{"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": text}]}], output_text=text)


def call_turn(name: str, call_id: str = "call-1") -> ModelTurn:
    return ModelTurn(output=[{"type": "function_call", "name": name,
                              "arguments": "{}", "call_id": call_id}], output_text="")


class ScriptedBackend:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.requests = []

    def respond(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        if not self.turns:
            raise AssertionError("test did not supply enough model turns")
        return self.turns.pop(0)


class RecordingTools:
    def __init__(self):
        self.calls = []

    @property
    def specs(self):
        return [ToolSpec(name, name, {"type": "object", "properties": {},
                                     "additionalProperties": False})
                for name in ("read", "edit")]

    def invoke(self, name, arguments=None):
        self.calls.append((name, arguments))
        return {"ok": True, "tool": name, "result": {"path": "evidence/example.json", "value": 1}}


class AgentStrategyTests(unittest.TestCase):
    def setUp(self):
        self.tools = RecordingTools()

    def test_plan_passes_step_results_even_when_backend_has_no_output_items(self):
        backend = ScriptedBackend(*[
            ModelTurn(output=[], output_text=text) for text in (
                json.dumps({"steps": ["inspect", "compare"]}),
                "first step observation", "second step finding", "final synthesis",
            )
        ])
        result = PlanExecuteAgent(self.tools, backend).run("Compare observations")
        second_step = backend.requests[2]["input_items"][-1]["content"]
        synthesis = backend.requests[3]["input_items"][-1]["content"]
        self.assertIn("first step observation", second_step)
        self.assertIn("inspect", second_step)
        self.assertIn("first step observation", synthesis)
        self.assertIn("second step finding", synthesis)
        self.assertEqual((result["status"], result["answer"], result["steps"]),
                         ("completed", "final synthesis", 4))

    def test_react_retains_observation_loop_and_legacy_result_shape(self):
        backend = ScriptedBackend(call_turn("read"), text_turn("See evidence/example.json"))
        result = ReActAgent(self.tools, backend).run("Read evidence", allowed_tools={"read"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual((result["steps"], result["tool_calls"]), (2, 1))
        self.assertEqual(result["evidence_paths"], ["evidence/example.json"])
        output = backend.requests[1]["input_items"][-1]
        self.assertEqual(output["call_id"], "call-1")
        self.assertEqual(json.loads(output["output"])["result"]["value"], 1)
        self.assertEqual(self.tools.calls, [("read", {})])

    def test_simple_never_exposes_or_executes_tools(self):
        backend = ScriptedBackend(call_turn("edit"))
        result = SimpleAgent(self.tools, backend).run("Answer directly")
        self.assertEqual(result["status"], "unexpected_tool_call")
        self.assertEqual(backend.requests[0]["tools"], [])
        self.assertEqual(result["tool_calls"], 0)
        self.assertEqual(self.tools.calls, [])

    def test_simple_text_and_empty_response(self):
        for text, expected in (("hello", "completed"), ("", "empty_response")):
            with self.subTest(text=text):
                result = SimpleAgent(self.tools, ScriptedBackend(text_turn(text))).run("Hello")
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["answer"], text)
                self.assertEqual(result["steps"], 1)

    def test_plan_executes_in_order_then_synthesizes_one_trace(self):
        backend = ScriptedBackend(
            text_turn('{"steps":["Inspect evidence", "Compare observations"]}'),
            call_turn("read"), text_turn("First finding"), text_turn("Second finding"),
            text_turn("Final answer: evidence/example.json"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.jsonl"
            result = PlanExecuteAgent(self.tools, backend, trace_path=path).run(
                "Compare evidence", allowed_tools={"read"})
            events = load_trace(path)
            self.assertTrue(replay_trace(path, ["read"])["trace_valid"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"], 5)
        self.assertEqual(len(result["step_results"]), 2)
        self.assertEqual(result["answer"], "Final answer: evidence/example.json")
        self.assertEqual(backend.requests[0]["tools"], [])
        self.assertEqual(backend.requests[-1]["tools"], [])
        self.assertIn("First finding", str(backend.requests[3]["input_items"]))
        self.assertEqual(sum(event["event"] == "run_started" for event in events), 1)
        self.assertEqual(sum(event["event"] == "run_finished" for event in events), 1)
        self.assertEqual(len({event["run_id"] for event in events}), 1)

    def test_invalid_plan_does_not_execute_any_tool(self):
        for response in ("not JSON", '{"steps":[]}', '{"steps":[""]}',
                         '{"steps":["one",2]}', '{"steps":["one","two"]}'):
            with self.subTest(response=response):
                backend = ScriptedBackend(text_turn(response))
                result = PlanExecuteAgent(self.tools, backend, max_plan_steps=1).run("Plan")
                self.assertEqual(result["status"], "invalid_plan")
                self.assertEqual(self.tools.calls, [])
                self.assertEqual(len(backend.requests), 1)

    def test_planner_and_synthesizer_cannot_execute_injected_calls(self):
        for turns in ((call_turn("edit"),),
                      (text_turn('{"steps":["Analyze"]}'), text_turn("finding"), call_turn("edit"))):
            with self.subTest(turn_count=len(turns)):
                result = PlanExecuteAgent(self.tools, ScriptedBackend(*turns)).run("Plan")
                self.assertEqual(result["status"], "unexpected_tool_call")
                self.assertEqual(self.tools.calls, [])

    def test_plan_stops_at_failed_step_and_counts_planning_against_budget(self):
        backend = ScriptedBackend(text_turn('{"steps":["one", "two"]}'), text_turn(""))
        result = PlanExecuteAgent(self.tools, backend).run("Plan")
        self.assertEqual(result["status"], "empty_response")
        self.assertEqual(len(result["step_results"]), 1)
        backend = ScriptedBackend(text_turn('{"steps":["one", "two"]}'), text_turn("first"))
        result = PlanExecuteAgent(self.tools, backend, max_steps=2).run("Plan")
        self.assertEqual(result["status"], "step_limit_exceeded")
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(backend.requests), 2)

    def test_reflection_critiques_revises_and_preserves_permissions(self):
        backend = ScriptedBackend(
            text_turn("Initial draft"),
            text_turn('{"approved":false,"feedback":"Read actual evidence before answering"}'),
            call_turn("edit"), text_turn("Revised answer"),
            text_turn('{"approved":true,"feedback":"Supported"}'),
        )
        result = ReflectionAgent(self.tools, backend, max_reflections=2).run(
            "Analyze", allowed_tools={"read"})
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["review_approved"])
        self.assertEqual(result["answer"], "Revised answer")
        self.assertEqual(result["steps"], 5)
        self.assertEqual(len(result["reflections"]), 2)
        self.assertEqual(backend.requests[1]["tools"], [])
        self.assertEqual(backend.requests[4]["tools"], [])
        rejection = json.loads(backend.requests[3]["input_items"][-1]["output"])
        self.assertEqual(rejection["error"]["code"], "TOOL_NOT_ALLOWED")
        self.assertEqual(self.tools.calls, [])

    def test_reflection_approval_returns_draft_without_refinement(self):
        backend = ScriptedBackend(text_turn("Draft"), text_turn('{"approved":true,"feedback":""}'))
        result = ReflectionAgent(self.tools, backend, max_reflections=3).run("Analyze")
        self.assertTrue(result["review_approved"])
        self.assertEqual(result["answer"], "Draft")
        self.assertEqual(len(backend.requests), 2)

    def test_reflection_budget_preserves_last_valid_draft(self):
        backend = ScriptedBackend(text_turn("Draft"), text_turn('{"approved":false,"feedback":"Improve"}'))
        result = ReflectionAgent(self.tools, backend, max_steps=2).run("Analyze")
        self.assertEqual(result["status"], "step_limit_exceeded")
        self.assertEqual(result["answer"], "Draft")
        self.assertFalse(result["review_approved"])
        self.assertEqual(len(backend.requests), 2)

    def test_refinement_limit_does_not_claim_last_draft_was_approved(self):
        backend = ScriptedBackend(text_turn("Draft"), text_turn('{"approved":false,"feedback":"Improve"}'),
                                  text_turn("Revision"))
        result = ReflectionAgent(self.tools, backend).run("Analyze")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer"], "Revision")
        self.assertFalse(result["review_approved"])
        self.assertTrue(result["reflection_limit_reached"])

    def test_invalid_critique_is_not_silently_treated_as_approval(self):
        for response in ('{"approved":"yes","feedback":""}', '{"approved":false,"feedback":""}',
                         "not JSON"):
            with self.subTest(response=response):
                backend = ScriptedBackend(text_turn("Draft"), text_turn(response))
                result = ReflectionAgent(self.tools, backend).run("Analyze")
                self.assertEqual(result["status"], "invalid_reflection")
                self.assertEqual(result["answer"], "Draft")
                self.assertFalse(result["review_approved"])

    def test_factory_supports_existing_task_modes_and_rejects_unknown_names(self):
        for name, expected in (("single", ReActAgent), ("react", ReActAgent),
                               ("simple", SimpleAgent), ("plan_execute", PlanExecuteAgent),
                               ("reflection", ReflectionAgent),
                               ("reviewer", ReflectionAgent)):
            self.assertIsInstance(create_agent(name, self.tools, ScriptedBackend()), expected)
        with self.assertRaises(ValueError):
            create_agent("missing", self.tools, ScriptedBackend())


if __name__ == "__main__":
    unittest.main()
