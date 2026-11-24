"""Capability isolation, blind evidence review, durable budgets and fail-closed audits."""
import json
from pathlib import Path
import tempfile
import unittest

from reclab_agent.agents import ResearchAuditAgent, create_agent
from reclab_agent.core.llm import ModelTurn
from reclab_agent.tools.audit import AuditToolbox
from tests.test_agent_strategies import RecordingTools, ScriptedBackend, call_turn, text_turn


def judgment(verdict="approved", refs=None):
    return text_turn(json.dumps({"verdict": verdict, "summary": "Checked observed facts.",
        "findings": [{"message": "Recorded source and score.", "evidence_ids": ["e1"] if refs is None else refs}]}))


class ResearchAuditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "run.audit.json"
        self.tools = RecordingTools()

    def agent(self, backend, **options):
        return ResearchAuditAgent(self.tools, backend, checkpoint_path=self.path,
                                  model_retries=0, **options)

    def test_contexts_are_separate_and_claim_arrives_only_after_blind_assessment(self):
        researcher = ScriptedBackend(call_turn("read"), text_turn("PRIVATE_RESEARCH_CLAIM"))
        auditor = ScriptedBackend(text_turn("INDEPENDENT_ASSESSMENT"), judgment())
        agent = self.agent(researcher, audit_backend=auditor)
        result = agent.run("Inspect the candidate.", allowed_tools={"read"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_requests"], 4)
        first = json.dumps(auditor.requests[0]["input_items"])
        self.assertNotIn("PRIVATE_RESEARCH_CLAIM", first)
        self.assertNotIn('"role": "assistant"', first)
        self.assertIn("PRIVATE_RESEARCH_CLAIM", json.dumps(auditor.requests[1]["input_items"]))
        self.assertNotIn("INDEPENDENT_ASSESSMENT", json.dumps(agent.get_history()))
        self.assertEqual([t["name"] for t in auditor.requests[0]["tools"]], ["read_audit_evidence"])
        self.assertEqual([t["name"] for t in auditor.requests[1]["tools"]], ["submit_audit"])
        self.assertIn("Tasks\n", researcher.requests[0]["instructions"])
        self.assertIn("Output\n", researcher.requests[0]["instructions"])
        self.assertIn("Requirements\n", auditor.requests[0]["instructions"])
        self.assertIn("Output example\n", auditor.requests[1]["input_items"][-1]["content"])
        state = json.loads(self.path.read_text())
        self.assertNotEqual(state["sessions"]["research"]["run_id"], state["sessions"]["audit"]["run_id"])

    def test_auditor_injected_mutation_never_dispatches_to_research_tools(self):
        researcher = ScriptedBackend(call_turn("read"), text_turn("PRIVATE_RESEARCH_CLAIM"))
        auditor = ScriptedBackend(call_turn("edit"), text_turn("inspected"), judgment())
        result = self.agent(researcher, audit_backend=auditor).run("inspect", allowed_tools={"read"})
        self.assertEqual(result["status"], "completed")
        self.assertEqual([name for name, _ in self.tools.calls], ["read"])
        self.assertIn("TOOL_NOT_ALLOWED", json.dumps(auditor.requests[1]["input_items"]))

    def test_researcher_cannot_borrow_audit_or_unapproved_tools(self):
        backend = ScriptedBackend(call_turn("edit"), text_turn("claim"), text_turn("missing"),
                                  judgment("insufficient_evidence", []))
        result = self.agent(backend).run("inspect", allowed_tools={"read"})
        self.assertEqual(self.tools.calls, [])
        self.assertEqual(result["status"], "audit_inconclusive")

    def test_audit_packet_cannot_be_mutated_and_paths_grant_no_access(self):
        original = {"observations": [{"id": "e1", "result": 1}]}
        tools = AuditToolbox(original)
        original["observations"].clear()
        returned = tools.invoke("read_audit_evidence")
        returned["result"]["observations"].clear()
        self.assertEqual(len(tools.invoke("read_audit_evidence")["result"]["observations"]), 1)
        self.assertFalse(tools.invoke("read_audit_evidence", {"path": "../secret"})["ok"])
        self.assertFalse(tools.invoke("run_experiment")["ok"])

    def test_rejection_is_not_completed_or_approved(self):
        backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("problem"), judgment("rejected"))
        result = self.agent(backend).run("inspect")
        self.assertEqual(result["status"], "audit_rejected")
        self.assertFalse(result["audit_approved"])
        self.assertEqual(result["answer"], "claim")

    def test_unknown_citations_and_invalid_shapes_fail_closed(self):
        for response in (judgment(refs=["e999"]), text_turn('{"verdict":"approved"}'), judgment(refs=[])):
            with self.subTest(response=response):
                backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("assessment"), response)
                agent = ResearchAuditAgent(self.tools, backend, model_retries=0, max_steps=4)
                self.assertEqual(agent.run("inspect")["status"], "invalid_audit")

    def test_submit_audit_is_data_never_a_business_tool(self):
        turn = judgment()
        submitted = ModelTurn([{"type": "function_call", "call_id": "audit", "name": "submit_audit",
                               "arguments": turn.output_text}], "")
        backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("assessment"), submitted)
        result = self.agent(backend).run("inspect")
        self.assertTrue(result["audit_approved"])
        self.assertEqual(len(self.tools.calls), 1)

    def test_trace_has_distinct_runs_one_task_and_finished_audit(self):
        backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("assessment"), judgment())
        trace = self.path.with_suffix(".jsonl")
        self.agent(backend, trace_path=trace).run("inspect")
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len({e["task_id"] for e in events}), 1)
        self.assertEqual(len({e["run_id"] for e in events}), 2)
        self.assertEqual(sum(e["event"] == "run_finished" for e in events), 2)

    def test_shared_budget_reserves_audit_capacity_and_never_resets(self):
        backend = ScriptedBackend(call_turn("read"), call_turn("read"))
        result = self.agent(backend, max_steps=4).run("inspect")
        self.assertEqual(result["status"], "step_limit_exceeded")
        self.assertFalse(result["audit_approved"])
        self.assertEqual(len(backend.requests), 2)
        self.assertEqual(result["model_requests"], 2)

    def test_failed_provider_request_remains_reserved_on_resume(self):
        researcher = ScriptedBackend(call_turn("read"), text_turn("PRIVATE_RESEARCH_CLAIM"))
        class Broken:
            def respond(self, **kwargs):
                raise TimeoutError("offline fault")
        with self.assertRaises(TimeoutError):
            self.agent(researcher, audit_backend=Broken()).run("inspect")
        self.assertEqual(json.loads(self.path.read_text())["requests"], 3)
        resume = ScriptedBackend(text_turn("assessment"), judgment())
        result = self.agent(ScriptedBackend(), audit_backend=resume).run("inspect")
        self.assertEqual(result["model_requests"], 5)
        self.assertEqual(len(self.tools.calls), 1)
        self.assertNotIn("PRIVATE_RESEARCH_CLAIM", json.dumps(resume.requests[0]["input_items"]))

    def test_finished_checkpoint_is_idempotent_and_scope_changes_are_rejected(self):
        backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("assessment"), judgment())
        agent = self.agent(backend)
        first = agent.run("inspect", allowed_tools={"read"})
        self.assertEqual(agent.run("inspect", allowed_tools={"read"}), first)
        self.assertEqual(len(backend.requests), 4)
        for kwargs in ({"prompt": "another", "allowed_tools": {"read"}},
                       {"prompt": "inspect", "allowed_tools": {"edit"}}):
            with self.assertRaises(ValueError):
                agent.run(**kwargs)
        from unittest.mock import patch
        with patch("reclab_agent.agents.research_audit_agent.ROLE_PROMPT_HASH", "changed-role-prompts"):
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                agent.run("inspect", allowed_tools={"read"})
        self.assertEqual(len(backend.requests), 4)

    def test_changed_frozen_evidence_is_rejected_before_reusing_audit(self):
        backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("assessment"), judgment())
        agent = self.agent(backend)
        agent.run("inspect")
        state = json.loads(self.path.read_text())
        state["packet"]["observations"][0]["result"]["result"]["value"] = 99
        self.path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "frozen audit evidence changed"):
            agent.run("inspect")
        self.assertEqual(len(backend.requests), 4)

    def test_pending_mutation_requires_reconciliation_without_replay(self):
        class BrokenTools(RecordingTools):
            def invoke(self, name, arguments=None):
                raise OSError("uncertain mutation outcome")
        backend = ScriptedBackend(call_turn("edit"))
        agent = ResearchAuditAgent(BrokenTools(), backend, checkpoint_path=self.path, model_retries=0)
        with self.assertRaises(OSError):
            agent.run("inspect")
        resumed = self.agent(ScriptedBackend()).run("inspect")
        self.assertEqual(resumed["status"], "reconciliation_required")
        self.assertFalse(resumed["audit_approved"])
        self.assertEqual(self.tools.calls, [])

    def test_inspection_cannot_exhaust_final_judgment_and_pass(self):
        backend = ScriptedBackend(call_turn("read"), text_turn("claim"), call_turn("read_audit_evidence"))
        result = self.agent(backend, max_steps=4).run("inspect")
        self.assertEqual(result["status"], "audit_incomplete")
        self.assertFalse(result["audit_approved"])

    def test_factory_and_minimum_budget(self):
        self.assertIsInstance(create_agent("research_audit", self.tools, ScriptedBackend()), ResearchAuditAgent)
        with self.assertRaises(ValueError):
            create_agent("team", self.tools, ScriptedBackend())
        with self.assertRaises(ValueError):
            ResearchAuditAgent(self.tools, ScriptedBackend(), max_steps=2)
