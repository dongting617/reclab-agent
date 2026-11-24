"""Role permissions, bounded review and durable recovery without remote services."""
import json
from pathlib import Path
import tempfile
import unittest

from reclab_agent.agents import TeamAgent, create_agent
from reclab_agent.core.llm import parse_json_response
from reclab_agent.evaluation.traces import replay_trace
from tests.test_agent_strategies import RecordingTools, ScriptedBackend, call_turn, text_turn


class TeamTests(unittest.TestCase):
    def test_structured_responses_accept_one_fence_but_reject_extra_prose(self):
        self.assertEqual(parse_json_response('```json\n{"steps":["read"]}\n```'), {"steps": ["read"]})
        self.assertEqual(parse_json_response('{"approved":true}'), {"approved": True})
        with self.assertRaises(ValueError):
            parse_json_response('Explanation\n```json\n{}\n```')
        with self.assertRaises(ValueError):
            parse_json_response('```json\n{}\n```\n```json\n{}\n```')

    def setUp(self):
        self.tools = RecordingTools()
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.checkpoint = Path(temp.name) / "roles.json"

    def agent(self, backend, **kwargs):
        return TeamAgent(self.tools, backend, max_steps=10, model_retries=0,
                         checkpoint_path=self.checkpoint, **kwargs)

    def test_roles_share_tools_and_budget_but_planner_reviewer_have_no_tools(self):
        backend = ScriptedBackend(text_turn('{"steps":["read evidence"]}'), call_turn("read"),
                                  text_turn("Observed value 1"), text_turn('{"approved":true,"feedback":""}'))
        agent = create_agent("team", self.tools, backend, max_steps=10, checkpoint_path=self.checkpoint)
        result = agent.run("inspect", allowed_tools={"read"})
        self.assertTrue(result["review_approved"])
        self.assertEqual(len(self.tools.calls), 1)
        self.assertEqual(result["model_requests"], 4)
        self.assertEqual([x["name"] for x in backend.requests[0]["tools"]], ["role_plan"])
        self.assertEqual([x["name"] for x in backend.requests[-1]["tools"]], ["role_review"])
        self.assertEqual([x["name"] for x in backend.requests[1]["tools"]], ["read"])
        with self.assertRaises(ValueError):
            self.agent(ScriptedBackend()).run("changed", allowed_tools={"read"})

    def test_restart_after_execution_only_retries_the_unfinished_reviewer(self):
        trace = self.checkpoint.with_suffix(".jsonl")
        class Failure(ScriptedBackend):
            def respond(self, **kwargs):
                if not self.turns:
                    raise RuntimeError("provider temporarily unavailable")
                return super().respond(**kwargs)
        first = Failure(text_turn('{"steps":["read"]}'), call_turn("read"), text_turn("value 1"))
        with self.assertRaises(RuntimeError):
            self.agent(first, trace_path=trace).run("inspect", allowed_tools={"read"})
        self.assertEqual(len(self.tools.calls), 1)
        review = ScriptedBackend(text_turn('{"approved":true,"feedback":""}'))
        result = self.agent(review, trace_path=trace).run("inspect", allowed_tools={"read"})
        self.assertEqual(len(review.requests), 1)
        self.assertEqual(len(self.tools.calls), 1)
        self.assertEqual(result["model_requests"], 5)  # Failed request remains charged.
        self.assertEqual(result["status"], "completed")
        again = self.agent(ScriptedBackend(), trace_path=trace).run("inspect", allowed_tools={"read"})
        self.assertEqual(again["answer"], "value 1")
        self.assertTrue(replay_trace(trace, ["read"])["trace_valid"])

    def test_review_has_at_most_one_revision(self):
        backend = ScriptedBackend(text_turn('{"steps":["read"]}'), text_turn("draft"),
                                  text_turn('{"approved":false,"feedback":"cite evidence"}'),
                                  text_turn("revision"), text_turn('{"approved":false,"feedback":"still missing"}'))
        result = self.agent(backend).run("inspect")
        self.assertEqual(result["revisions"], 1)
        self.assertFalse(result["review_approved"])
        self.assertEqual(result["model_requests"], 5)

    def test_reviewer_tool_request_never_executes(self):
        backend = ScriptedBackend(text_turn('{"steps":["read"]}'), text_turn("draft"), call_turn("edit"))
        result = self.agent(backend).run("inspect")
        self.assertEqual(result["status"], "unexpected_tool_call")
        self.assertEqual(self.tools.calls, [])

    def test_structured_role_submissions_are_data_and_not_business_tool_calls(self):
        from reclab_agent.core import ModelTurn
        def submission(name, arguments):
            return ModelTurn(output=[{"type": "function_call", "name": name,
                                      "arguments": json.dumps(arguments), "call_id": name}], output_text="")
        backend = ScriptedBackend(submission("role_plan", {"steps": ["read"]}), call_turn("read"),
                                  text_turn("value 1"), submission("role_review", {"approved": True, "feedback": ""}))
        result = self.agent(backend).run("inspect")
        self.assertTrue(result["review_approved"])
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual([name for name, _ in self.tools.calls], ["read"])

    def test_uncertain_tool_side_effect_is_not_replayed_on_resume(self):
        class CrashTools(RecordingTools):
            def invoke(self, name, arguments=None):
                super().invoke(name, arguments)
                raise RuntimeError("crash after side effect")
        self.tools = CrashTools()
        with self.assertRaises(RuntimeError):
            self.agent(ScriptedBackend(text_turn('{"steps":["read"]}'), call_turn("read"))).run("inspect")
        result = self.agent(ScriptedBackend()).run("inspect")
        self.assertEqual(result["status"], "reconciliation_required")
        self.assertEqual(len(self.tools.calls), 1)

    def test_all_roles_consume_one_limit(self):
        backend = ScriptedBackend(text_turn('{"steps":["read"]}'), text_turn("draft"))
        result = TeamAgent(self.tools, backend, max_steps=2, model_retries=0).run("inspect")
        self.assertEqual(result["status"], "step_limit_exceeded")
        self.assertEqual(result["model_requests"], 2)
