"""Replay observed malformed submissions; check bounded repair and citations offline."""
import json
from pathlib import Path
import tempfile
import unittest

from reclab_agent.agents import ResearchAuditAgent
from reclab_agent.core.llm import ModelTurn
from reclab_agent.evaluation.diagnostics import grade_answer, run_diagnostic_suite
from reclab_agent.tools.citation_validation import check_claim_citations
from reclab_agent.tools.evidence import json_pointer
from tests.test_agent_strategies import RecordingTools, ScriptedBackend, call_turn, text_turn
from tests.test_research_audit_agent import judgment

ROOT = Path(__file__).resolve().parents[1]


def observation(document, base=""):
    return {"id": "e1", "tool": "read_json_values", "result": {"ok": True,
        "result": {"path": "evidence/case.json", "values": {base: document}}}}


def claim(pointer="/metrics/ndcg@20", decision="retain"):
    return text_turn(json.dumps({"decision": decision, "reason": "Observed evidence",
        "citations": [{"path": "evidence/case.json", "pointer": pointer}]}))


class EvidenceTools(RecordingTools):
    def invoke(self, name, arguments=None):
        self.calls.append((name, arguments))
        return observation({"metrics": {"ndcg@20": 0.1}})["result"]

    @property
    def specs(self):
        from reclab_agent.tools.contracts import ToolSpec
        return [ToolSpec("read_json_values", "Read fixture evidence", {"type": "object", "properties": {}})]


class AuditProtocolTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name)/"run.audit.json"
        self.tools = RecordingTools()

    def agent(self, backend, **kwargs):
        return ResearchAuditAgent(self.tools, backend, model_retries=0, checkpoint_path=self.path, **kwargs)

    def test_actual_invalid_tool_arguments_are_resubmitted_once_as_valid_data(self):
        cases = json.loads((ROOT/"tests/fixtures/audit-malformed-submissions-v1.json").read_text(encoding="utf-8"))["cases"]
        for case in cases:
            with self.subTest(case=case["id"]):
                self.path.unlink(missing_ok=True)
                malformed = ModelTurn([{"type": "function_call", "call_id": "bad-audit", "name": "submit_audit",
                                        "arguments": case["arguments"]}], "")
                backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("assessment"), malformed, judgment())
                result = self.agent(backend, max_steps=5).run("inspect")
                self.assertTrue(result["audit_approved"])
                self.assertEqual(result["audit_repairs"], 1)
                self.assertEqual(result["model_requests"], 5)
                feedback = next(i for i in backend.requests[-1]["input_items"] if i.get("type") == "function_call_output" and i.get("call_id") == "bad-audit")
                self.assertEqual(json.loads(feedback["output"])["error"]["code"], "INVALID_AUDIT_JSON")
                self.assertIn(case["arguments"], [i.get("arguments") for i in json.loads(self.path.read_text())["sessions"]["audit"]["inputs"]])

    def test_second_invalid_submission_stops_without_extra_requests(self):
        bad = text_turn('{"verdict":"approved"}')
        backend = ScriptedBackend(call_turn("read"), text_turn("claim"), text_turn("assessment"), bad, bad)
        result = self.agent(backend).run("inspect")
        self.assertEqual(result["status"], "invalid_audit")
        self.assertFalse(result["audit_approved"])
        self.assertEqual(result["model_requests"], 5)
        self.assertEqual(len(self.tools.calls), 1)

    def test_repair_timeout_does_not_reset_budget_or_repair_count(self):
        class Broken(ScriptedBackend):
            def respond(inner, **kwargs):
                if len(inner.requests) == 4:
                    raise TimeoutError("offline repair timeout")
                return super().respond(**kwargs)
        backend = Broken(call_turn("read"), text_turn("claim"), text_turn("assessment"), text_turn("bad JSON"))
        with self.assertRaises(TimeoutError):
            self.agent(backend, max_steps=6).run("inspect")
        saved = json.loads(self.path.read_text())
        self.assertEqual((saved["requests"], saved["audit_repairs"]), (5, 1))
        resumed = self.agent(ScriptedBackend(judgment()), max_steps=6).run("inspect")
        self.assertEqual((resumed["model_requests"], resumed["audit_repairs"]), (6, 1))
        self.assertTrue(resumed["audit_approved"])

    def test_citation_repair_has_no_tools_and_does_not_leak_into_blind_context(self):
        tools = EvidenceTools()
        main = ScriptedBackend(call_turn("read_json_values"), claim("/"), claim())
        auditor = ScriptedBackend(text_turn("assessment"), judgment())
        result = ResearchAuditAgent(tools, main, audit_backend=auditor, max_steps=5,
            model_retries=0, checkpoint_path=self.path).run("inspect")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["citation_check"]["status"], "passed")
        self.assertEqual(result["citation_repairs"], 1)
        self.assertEqual(main.requests[-1]["tools"], [])
        self.assertNotIn("host_citation_check", json.dumps(auditor.requests[0]["input_items"]))
        self.assertIn("host_citation_check", json.dumps(auditor.requests[1]["input_items"]))

    def test_bad_citations_cannot_be_approved_or_repaired_by_mutation(self):
        for corrected in (claim("/"), claim(decision="reject"), call_turn("edit"), text_turn("not JSON")):
            with self.subTest(corrected=corrected):
                tools = EvidenceTools()
                main = ScriptedBackend(call_turn("read_json_values"), claim("/"), corrected)
                auditor = ScriptedBackend()
                result = ResearchAuditAgent(tools, main, audit_backend=auditor, max_steps=5, model_retries=0).run("inspect")
                self.assertEqual(result["status"], "invalid_citations")
                self.assertFalse(result["audit_approved"])
                self.assertEqual(len(tools.calls), 1)
                self.assertEqual(auditor.requests, [])

    def test_citation_repair_cannot_borrow_reserved_audit_budget(self):
        main = ScriptedBackend(call_turn("read_json_values"), claim("/"))
        result = ResearchAuditAgent(EvidenceTools(), main, max_steps=4, model_retries=0).run("inspect")
        self.assertEqual(result["status"], "invalid_citations")
        self.assertEqual(result["citation_repairs"], 0)
        self.assertEqual(result["model_requests"], 2)

    def test_citation_repair_resumes_without_replaying_tools_or_resetting_allowance(self):
        class Broken(ScriptedBackend):
            def respond(inner, **kwargs):
                if len(inner.requests) == 2:
                    raise TimeoutError("offline citation correction timeout")
                return super().respond(**kwargs)
        tools = EvidenceTools()
        backend = Broken(call_turn("read_json_values"), claim("/"))
        agent = ResearchAuditAgent(tools, backend, checkpoint_path=self.path, model_retries=0, max_steps=6)
        with self.assertRaises(TimeoutError):
            agent.run("inspect")
        saved = json.loads(self.path.read_text())
        self.assertEqual((saved["phase"], saved["requests"], saved["citation_repairs"]), ("cite_repair", 3, 1))
        resumed = ResearchAuditAgent(tools, ScriptedBackend(claim(), text_turn("assessment"), judgment()),
            checkpoint_path=self.path, model_retries=0, max_steps=6).run("inspect")
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual((resumed["model_requests"], resumed["citation_repairs"]), (6, 1))
        self.assertEqual(len(tools.calls), 1)


class CitationTests(unittest.TestCase):
    def test_pointer_root_empty_key_and_escapes_are_distinct(self):
        doc = {"": 7, "a/b": {"~x": [4]}}
        self.assertEqual(json_pointer(doc, ""), doc)
        self.assertEqual(json_pointer(doc, "/"), 7)
        self.assertEqual(json_pointer(doc, "/a~1b/~0x/0"), 4)
        for pointer in ("/~2", "/a~1b/~0x/00", "/a~1b/~0x/+0", "/a~1b/~0x/-1"):
            with self.subTest(pointer=pointer), self.assertRaises(ValueError):
                json_pointer(doc, pointer)

    def test_citations_use_only_observed_subtrees_and_reject_conflicting_values(self):
        obs = [observation({"ndcg@20": .1}, base="/metrics")]
        self.assertEqual(check_claim_citations(claim().output_text, obs)["status"], "passed")
        self.assertEqual(check_claim_citations(claim("").output_text, obs)["status"], "failed")
        self.assertEqual(check_claim_citations(claim().output_text, obs + [observation({"ndcg@20": .2}, base="/metrics")])["status"], "failed")
        bad = json.loads(claim().output_text)
        bad["citations"][0]["path"] = "../secret.json"
        self.assertEqual(check_claim_citations(json.dumps(bad), obs)["status"], "failed")

    def test_bad_citation_does_not_erase_a_correct_decision(self):
        task = {"evidence": {"metrics": {"ndcg@20": .1}}, "expected_decision": "retain", "required_pointer": "/metrics"}
        answer = json.loads(claim("/").output_text)
        graded = grade_answer(answer, task)
        self.assertTrue(graded["decision_correct"])
        self.assertFalse(graded["citations_valid"])
        self.assertFalse(graded["passed"])
        answer["citations"].append(json.loads(claim().output_text)["citations"][0])
        self.assertFalse(grade_answer(answer, task)["passed"])  # Every citation must be valid.

    def test_evaluator_reports_invalid_citation_separately(self):
        for pointer, error in (("/", "INVALID_CITATION"), (7, "INVALID_ANSWER_SCHEMA")):
            with self.subTest(pointer=pointer), tempfile.TemporaryDirectory() as directory:
                report = run_diagnostic_suite(ROOT/"tasks/diagnostic-dev.json", Path(directory)/"report.json",
                    mode="react", backend=ScriptedBackend(claim(pointer)), max_steps=1, max_requests=1)
            first = report["results"][0]
            self.assertTrue(first["decision_correct"])
            self.assertFalse(first["passed"])
            self.assertEqual(first["failure"]["code"], error)
