"""Offline trace correlation and recovery at explicit interruption boundaries."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from reclab_agent.agents import ReActAgent, ResearchAuditAgent
from reclab_agent.application.runs import RunCatalog
from reclab_agent.core.trace import TraceWriter, model_error_category
from reclab_agent.workflow.coordinator import StudyCoordinator
from reclab_agent.workflow.stages import StudyStage
from tests.test_agent_strategies import RecordingTools, ScriptedBackend, call_turn, text_turn
from tests.test_research_audit_agent import judgment


class NamedBackend(ScriptedBackend):
    model = "offline-research-model"


class AuditBackend(ScriptedBackend):
    model = "offline-audit-model"


class DurableTools(RecordingTools):
    def __init__(self, root):
        super().__init__()
        self.counter = Path(root) / "tools.log"

    def invoke(self, name, arguments=None):
        with self.counter.open("a", encoding="utf-8") as stream:
            stream.write(name + "\n")
        return super().invoke(name, arguments)


def interrupt_audit(root):
    """Child exits after persisting its third request reservation, before a reply."""
    class ExitBackend:
        model = AuditBackend.model

        def respond(self, **kwargs):
            os._exit(23)

    root = Path(root)
    ResearchAuditAgent(DurableTools(root), NamedBackend(call_turn("read"), text_turn("claim")),
        audit_backend=ExitBackend(), checkpoint_path=root / "audit.json", trace_path=root / "agent.jsonl",
        max_steps=8, model_retries=0, trace_context={"study_id": "study-1", "stage": "experiment"}).run("inspect")


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trace = self.root / "agent.jsonl"

    def events(self):
        return [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]

    def test_retry_has_distinct_requests_same_prompt_and_measured_outcomes(self):
        class Retrying(NamedBackend):
            attempts = 0

            def respond(self, **kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise TimeoutError("offline injected timeout")
                return text_turn("done")

        ReActAgent(RecordingTools(), Retrying(), trace_path=self.trace, model_retries=1,
            trace_context={"study_id": "s1", "parent_run_id": "s1", "stage": "analysis"}).run("inspect")
        events = self.events()
        starts = [e for e in events if e["event"] == "model_request_started"]
        error = next(e for e in events if e["event"] == "model_error")
        success = next(e for e in events if e["event"] == "model_turn")
        self.assertEqual(len(starts), 2)
        self.assertNotEqual(starts[0]["request_id"], starts[1]["request_id"])
        self.assertEqual(error["request_id"], starts[0]["request_id"])
        self.assertEqual(success["request_id"], starts[1]["request_id"])
        self.assertEqual(error["prompt_hash"], success["prompt_hash"])
        self.assertEqual(error["error_category"], "timeout")
        for event in (error, success):
            self.assertGreaterEqual(event["elapsed_ms"], 0)
            self.assertEqual(event["model"], NamedBackend.model)
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        for event in events:
            self.assertIsNotNone(datetime.fromisoformat(event["timestamp"]).tzinfo)
            self.assertEqual(event["study_id"], "s1")
            self.assertEqual(event["stage"], "analysis")

    def test_request_hash_changes_with_instructions_input_and_tool_schema(self):
        tools = RecordingTools()
        agent = ReActAgent(tools, NamedBackend(text_turn("a"), text_turn("b"), text_turn("c")),
                           trace_path=self.trace)
        session = agent._session("same task")
        session.turn("first", use_tools=False)
        session.turn("second", use_tools=False)
        session.turn("second", use_tools=True)
        events = [e for e in self.events() if e["event"] == "model_turn"]
        self.assertNotEqual(events[0]["instructions_hash"], events[1]["instructions_hash"])
        self.assertNotEqual(events[0]["input_hash"], events[1]["input_hash"])
        self.assertNotEqual(events[1]["tool_schema_hash"], events[2]["tool_schema_hash"])
        self.assertEqual(len({e["prompt_hash"] for e in events}), 3)

    def test_error_categories_use_exception_types_and_status_not_message_text(self):
        class Rejected(RuntimeError):
            request_not_sent = True

        class ProviderError(Exception):
            def __init__(self, status):
                self.status_code = status

        for error, category in ((Rejected(), "local_rejection"), (ConnectionError(), "connection"),
                (ProviderError(429), "rate_limit"), (ProviderError(401), "authentication"),
                (ProviderError(503), "provider_server"), (ProviderError(400), "provider_request"),
                (json.JSONDecodeError("bad", "{", 0), "invalid_json"), (RuntimeError("timeout"), "unknown")):
            with self.subTest(category=category):
                self.assertEqual(model_error_category(error), category)

    def test_coordinator_stage_and_agent_tool_events_share_research_identity(self):
        class Tools(RecordingTools):
            active_stage = None
            active_candidate_id = None
            usage = {}

            def save_state(self):
                pass

            def invoke(self, name, arguments=None):
                return {"ok": True, "result": {"experiment_id": "exp-1"}}

        tools = Tools()
        backend = NamedBackend(call_turn("read"), text_turn("done"))

        def factory(research, model, **options):
            agent = ReActAgent(research, model, trace_writer=options["trace_writer"],
                               trace_context=options["trace_context"])
            return SimpleNamespace(run=lambda prompt, allowed_tools: agent.run(
                prompt, task_id="task-1", allowed_tools=allowed_tools))

        coordinator = StudyCoordinator(SimpleNamespace(output_directory=self.root,
            task=SimpleNamespace(task_id="task-1")), None, None, tools, backend, factory,
            {"context": {"target_model": "SASRec"}}, lambda *args: None)
        backend.calls, backend.limit = 0, 8
        coordinator.stage(StudyStage.EXPERIMENT, candidate_id="candidate-1")
        coordinator.agent("inspect", {"read"})
        coordinator.invoke("analyze_experiment", experiment_id="exp-1")
        events = self.events()
        self.assertEqual(events[0]["event"], "stage_changed")
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        self.assertTrue(all(e["study_id"] == self.root.name for e in events))
        self.assertTrue(all(e["task_id"] == "task-1" for e in events))
        tool_events = [e for e in events if e["event"] == "tool_call"]
        self.assertEqual([e["experiment_id"] for e in tool_events], ["exp-1", "exp-1"])
        agent_start = next(e for e in events if e["event"] == "run_started")
        self.assertEqual(agent_start["parent_run_id"], self.root.name)

    def test_process_exit_after_request_reservation_resumes_without_replaying_tool(self):
        child = subprocess.run([sys.executable, "-B", "-c",
            "from tests.test_observability import interrupt_audit; import sys; interrupt_audit(sys.argv[1])",
            str(self.root)], cwd=Path(__file__).absolute().parents[1], capture_output=True, timeout=30)
        self.assertEqual(child.returncode, 23, child.stderr.decode(errors="replace"))
        saved = json.loads((self.root / "audit.json").read_text())
        self.assertEqual(saved["requests"], 3)
        before = len(self.events())
        result = ResearchAuditAgent(DurableTools(self.root), NamedBackend(),
            audit_backend=AuditBackend(text_turn("assessment"), judgment()),
            checkpoint_path=self.root / "audit.json", trace_path=self.trace,
            max_steps=8, model_retries=0,
            trace_context={"study_id": "study-1", "stage": "experiment"}).run("inspect")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_requests"], 5)
        self.assertEqual((self.root / "tools.log").read_text().splitlines(), ["read"])
        events = self.events()
        self.assertEqual([e["seq"] for e in events], list(range(1, len(events) + 1)))
        self.assertGreater(events[before]["seq"], events[before - 1]["seq"])
        research_id = saved["sessions"]["research"]["run_id"]
        audit_events = [e for e in events if e["role"] == "auditor"]
        self.assertTrue(all(e["parent_run_id"] == research_id for e in audit_events))
        self.assertTrue(all(e["study_id"] == "study-1" for e in events))
        starts = {e["request_id"] for e in events if e["event"] == "model_request_started"}
        finished = {e["request_id"] for e in events if e["event"] in {"model_turn", "model_error"}}
        self.assertEqual(len(starts - finished), 1)  # The exited process never produced a reply.
        self.assertTrue(all(e["model"] == AuditBackend.model for e in audit_events))

    def test_interruption_after_possible_side_effect_requires_manual_reconciliation(self):
        class InterruptedTools(DurableTools):
            def invoke(self, name, arguments=None):
                super().invoke(name, arguments)
                raise KeyboardInterrupt("after side effect, before recording result")

        tools = InterruptedTools(self.root)
        options = {"checkpoint_path": self.root / "audit.json", "trace_path": self.trace,
                   "max_steps": 8, "model_retries": 0}
        with self.assertRaises(KeyboardInterrupt):
            ResearchAuditAgent(tools, NamedBackend(call_turn("read")), **options).run("inspect")
        self.assertTrue(json.loads((self.root / "audit.json").read_text())["pending_tools"])
        backend = NamedBackend()
        result = ResearchAuditAgent(tools, backend, **options).run("inspect")
        self.assertEqual(result["status"], "reconciliation_required")
        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(backend.requests, [])
        self.assertEqual((self.root / "tools.log").read_text().splitlines(), ["read"])

    def test_persisted_events_continue_after_catalog_and_writer_restart(self):
        session = self.root / "sessions" / "s1"
        session.mkdir(parents=True)
        (session / "report.json").write_text(json.dumps({"status": "completed", "context": {}, "stages": []}))
        path = session / "agent.jsonl"
        TraceWriter(path).write({"event": "first", "study_id": "s1"})
        catalog = RunCatalog(self.root, session_roots=[session.parent])
        cursor = catalog.events("study-s1")[-1]["seq"]
        TraceWriter(path).write({"event": "second", "study_id": "s1"})
        reopened = RunCatalog(self.root, session_roots=[session.parent])
        remaining = reopened.events("study-s1", cursor)
        self.assertEqual([e["seq"] for e in remaining], [2])
        self.assertEqual(remaining[0]["payload"]["seq"], 2)
        self.assertIsNotNone(remaining[0]["payload"]["timestamp"])

    def test_torn_event_does_not_get_silently_overwritten_or_appended(self):
        self.trace.write_text('{"event":"partial"', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete final event"):
            TraceWriter(self.trace)
        self.assertEqual(self.trace.read_text(), '{"event":"partial"')


if __name__ == "__main__":
    unittest.main()
