"""Provider adaptation and shared run behavior use the current core API."""
from __future__ import annotations
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import reclab_agent
from reclab_agent.agents import ReActAgent
from reclab_agent.core import Agent, AgentConfig
from reclab_agent.core.llm import ModelBackend, ModelTurn, OpenAIResponsesBackend
from reclab_agent.core.session import RunSession, StepLimitExceeded, function_tools
from reclab_agent.core.trace import TraceWriter
from reclab_agent.evaluation.traces import load_trace, replay_trace
from reclab_agent.tools import ToolRegistry, ToolSpec


def text_turn(text):
    return ModelTurn(output=[{"type": "message", "role": "assistant", "content": [
        {"type": "output_text", "text": text}]}], output_text=text)


class RecordingBackend:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def respond(self, **request):
        self.requests.append(copy.deepcopy(request))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class JsonObject:
    def __init__(self, payload):
        self.payload = payload
        self.dump_modes = []

    def model_dump(self, *, mode):
        self.dump_modes.append(mode)
        return copy.deepcopy(self.payload)


class AgentCoreTests(unittest.TestCase):
    def test_public_interfaces_export_current_implementations(self):
        self.assertIs(reclab_agent.Agent, Agent)
        self.assertIs(reclab_agent.AgentConfig, AgentConfig)
        self.assertIs(reclab_agent.ReActAgent, ReActAgent)
        self.assertIs(reclab_agent.ModelTurn, ModelTurn)
        self.assertIs(reclab_agent.ModelBackend, ModelBackend)
        self.assertIs(reclab_agent.OpenAIResponsesBackend, OpenAIResponsesBackend)
        self.assertIs(reclab_agent.ToolRegistry, ToolRegistry)
        self.assertIs(reclab_agent.ToolSpec, ToolSpec)

    def test_injected_responses_client_retains_request_and_conversion(self):
        output = JsonObject({"type": "function_call", "name": "read", "arguments": "{}", "call_id": "c1"})
        usage = JsonObject({"input_tokens": 12, "output_tokens": 4})
        calls = []

        def create(**request):
            calls.append(request)
            return SimpleNamespace(output=[output], output_text="answer", id="response-1", usage=usage)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        backend = OpenAIResponsesBackend("injected-model", client)
        inputs = [{"role": "user", "content": "question"}]
        tools = [{"type": "function", "name": "read"}]
        turn = backend.respond(instructions="custom instructions", input_items=inputs, tools=tools)

        self.assertEqual(calls, [{
            "model": "injected-model", "instructions": "custom instructions", "input": inputs,
            "tools": tools, "tool_choice": "auto", "parallel_tool_calls": False, "store": False,
        }])
        self.assertEqual(turn, ModelTurn([output.payload], "answer", "response-1", usage.payload))
        self.assertEqual(output.dump_modes, ["json"])
        self.assertEqual(usage.dump_modes, ["json"])

    def test_responses_without_text_or_usage_have_original_defaults(self):
        response = SimpleNamespace(output=[], output_text=None, id="empty", usage=None)
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: response))
        turn = OpenAIResponsesBackend("injected-model", client).respond(
            instructions="instructions", input_items=[], tools=[])
        self.assertEqual(turn, ModelTurn([], "", "empty", None))

    def test_history_is_a_deep_copy_and_each_run_starts_fresh(self):
        toolbox = ToolRegistry().register_function(
            ToolSpec("read", "Read evidence", {"type": "object", "properties": {},
                                              "additionalProperties": False}),
            lambda: {"path": "evidence/example.json"},
        )
        call = ModelTurn(output=[{"type": "function_call", "name": "read", "arguments": "{}",
                                  "call_id": "first-call"}], output_text="")
        backend = RecordingBackend(call, text_turn("first answer"), text_turn("second answer"))
        agent = ReActAgent(toolbox, backend)
        first = agent.run("first task")
        history = agent.get_history()
        snapshot = copy.deepcopy(history)
        history[-1]["content"][0]["text"] = "caller mutation"
        history[0]["content"] = "changed question"
        self.assertEqual(agent.get_history(), snapshot)
        second = agent.run("second task")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(second["steps"], 1)
        self.assertEqual(second["tool_calls"], 0)
        self.assertEqual(second["evidence_paths"], [])
        self.assertEqual(backend.requests[-1]["input_items"], [{"role": "user", "content": "second task"}])
        self.assertEqual(agent.get_history()[0]["content"], "second task")
        self.assertEqual(snapshot[-1]["content"][0]["text"], "first answer")
        agent.clear_history()
        self.assertEqual(agent.get_history(), [])

    def test_config_overrides_are_independent_and_use_custom_instructions(self):
        config = AgentConfig(max_steps=4, model_retries=3)
        backend = RecordingBackend(text_turn("answer"))
        agent = ReActAgent(ToolRegistry(), backend, config=config, max_steps=2,
                           name="custom", system_prompt="custom instructions")
        self.assertEqual(agent.config, AgentConfig(max_steps=2, model_retries=3))
        self.assertEqual(config, AgentConfig(max_steps=4, model_retries=3))
        self.assertEqual(agent.name, "custom")
        agent.run("question")
        self.assertEqual(backend.requests[0]["instructions"], "custom instructions")

        agent = ReActAgent(ToolRegistry(), RecordingBackend(), config=config, model_retries=0)
        self.assertEqual(agent.config, AgentConfig(max_steps=4, model_retries=0))

    def test_invalid_limits_and_prompt_fail_before_requesting_model(self):
        backend = RecordingBackend()
        for kwargs in ({"max_steps": 0}, {"max_steps": True}, {"max_steps": 1.5},
                       {"model_retries": -1}, {"model_retries": False}, {"model_retries": 1.5},
                       {"system_prompt": " "}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReActAgent(ToolRegistry(), backend, **kwargs)
        agent = ReActAgent(ToolRegistry(), backend)
        for prompt in (None, "", " \n"):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                agent.run(prompt)
        self.assertEqual(backend.requests, [])

    def test_shared_session_counts_retries_and_phases_in_one_run(self):
        backend = RecordingBackend(TimeoutError("temporary failure"), text_turn("plan"), text_turn("answer"))
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "session.jsonl"
            session = RunSession(ToolRegistry(), backend, AgentConfig(max_steps=2, model_retries=1),
                                 TraceWriter(trace), "instructions", "question", task_id="task-1")
            session.turn(use_tools=False, phase="plan")
            session.turn(use_tools=False, phase="execute")
            with self.assertRaises(StepLimitExceeded):
                session.turn()
            result = session.finish("completed", "answer")
            with self.assertRaises(RuntimeError):
                session.finish("completed", "another answer")
            attempts = len(backend.requests)
            with self.assertRaises(RuntimeError):
                session.turn()
            with self.assertRaises(RuntimeError):
                session.execute(text_turn("late response"))
            self.assertEqual(len(backend.requests), attempts)
            events = load_trace(trace)
            self.assertTrue(replay_trace(trace, [])["trace_valid"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(backend.requests), 3)
        self.assertEqual(backend.requests[0], backend.requests[1])
        self.assertEqual({event["run_id"] for event in events}, {result["run_id"]})
        self.assertEqual({event["task_id"] for event in events}, {"task-1"})
        self.assertEqual([event["phase"] for event in events if event["event"] == "model_turn"],
                         ["plan", "execute"])
        error = next(event for event in events if event["event"] == "model_error")
        self.assertEqual((error["step"], error["attempt"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
