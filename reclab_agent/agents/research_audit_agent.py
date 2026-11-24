"""A researcher and an independent auditor, with isolated sessions and capabilities."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from reclab_agent.core.agent import Agent
from reclab_agent.core.session import RunSession, StepLimitExceeded, function_calls
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.execution.locking import job_lock
from reclab_agent.infrastructure.files import atomic_write, read_regular_file
from reclab_agent.tools.audit import AUDIT_PROTOCOL_VERSION, AUDIT_SCHEMA, AuditToolbox, validate_audit
from reclab_agent.tools.citation_validation import check_claim_citations
from reclab_agent.agents.research_audit_prompts import (
    AUDIT_INSTRUCTIONS, AUDIT_INSPECTION_PROMPT, AUDIT_JUDGMENT_PROMPT,
    AUDIT_REPAIR_PROMPT, CITATION_REPAIR_PROMPT, RESEARCHER_INSTRUCTIONS, ROLE_PROMPT_HASH,
)


class ResearchAuditAgent(Agent):
    """Serial independent audit with bounded data corrections, no experimental revision."""

    def __init__(self, *args, checkpoint_path=None, audit_backend=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.max_steps < 3:
            raise ValueError("research_audit requires at least three shared model requests")
        self.audit_backend = audit_backend or self.backend
        conversation = getattr(self.toolbox, "conversation_path", None) or self.trace.path
        self.checkpoint_path = (Path(checkpoint_path) if checkpoint_path else
                                Path(str(conversation) + ".audit.json") if conversation else None)

    def run(self, prompt, *, task_id=None, allowed_tools=None):
        if self.checkpoint_path is None:
            return self._run(prompt, task_id=task_id, allowed_tools=allowed_tools)
        with job_lock(self.checkpoint_path.parent, self.checkpoint_path.name) as acquired:
            if not acquired:
                raise ValueError("another controller owns this research audit checkpoint")
            return self._run(prompt, task_id=task_id, allowed_tools=allowed_tools)

    def _run(self, prompt, *, task_id, allowed_tools):
        names = {s.name for s in self.toolbox.specs} if allowed_tools is None else set(allowed_tools)
        scope = content_hash({"version": AUDIT_PROTOCOL_VERSION, "prompt": prompt, "task_id": task_id,
            "max_steps": self.max_steps, "retries": self.model_retries, "instructions": self.system_prompt,
            "audit_instructions": AUDIT_INSTRUCTIONS, "tools": sorted(names),
            "role_prompt_hash": ROLE_PROMPT_HASH,
            "audit_schema": AUDIT_SCHEMA,
            "model": getattr(self.backend, "model", None), "audit_model": getattr(self.audit_backend, "model", None)})
        state = {"scope": scope, "phase": "research", "requests": 0, "pending_tools": False,
                 "answer": "", "observations": [], "sessions": {}}
        if self.checkpoint_path and self.checkpoint_path.exists():
            state = json.loads(read_regular_file(self.checkpoint_path, max_bytes=32_000_000))
            if state.get("scope") != scope:
                raise ValueError("audit checkpoint does not match task, models, permissions or budget")
            if "packet" in state and content_hash(state["packet"]) != state.get("packet_hash"):
                raise ValueError("frozen audit evidence changed")
            if "result" in state:
                self._history = state["sessions"]["research"]["inputs"]
                return state["result"]
        sessions = {}

        def save():
            for role, session in sessions.items():
                state["sessions"][role] = {"inputs": session.inputs, "steps": session.steps,
                    "tool_calls": session.tool_calls, "evidence": sorted(session.evidence), "run_id": session.run_id,
                    "experiment_id": session.experiment_id}
            if self.checkpoint_path:
                atomic_write(self.checkpoint_path, canonical_json(state).encode("utf-8"))

        def make_session(role, toolbox, backend, instructions, request, permissions):
            saved = state["sessions"].get(role, {})
            class BoundedBackend:
                model = getattr(backend, "model", None)

                def respond(inner, **kwargs):
                    # Research leaves two requests for blind inspection and final judgment.
                    limit = self.max_steps - 2 if state["phase"] in {"research", "cite_repair"} else self.max_steps
                    if state["requests"] >= limit:
                        raise StepLimitExceeded
                    state["requests"] += 1
                    save()  # Failed provider attempts remain charged across restart.
                    return backend.respond(**kwargs)
            context = {**self.trace_context, "role": "auditor" if role == "audit" else "researcher"}
            if role == "audit":
                context.update(study_id=sessions["research"].trace_context["study_id"],
                               parent_run_id=sessions["research"].run_id,
                               experiment_id=sessions["research"].experiment_id)
            if saved.get("experiment_id") is not None:
                context["experiment_id"] = saved["experiment_id"]
            session = RunSession(toolbox, BoundedBackend(), self.config, self.trace, instructions, request,
                task_id=task_id or (sessions["research"].task_id if role == "audit" else None),
                allowed_tools=permissions, resume_run_id=saved.get("run_id"), trace_context=context)
            if saved:
                session.inputs = saved["inputs"]
                session.steps, session.tool_calls = saved["steps"], saved["tool_calls"]
                session.evidence = set(saved["evidence"])
                session.experiment_id = saved.get("experiment_id", session.experiment_id)
            sessions[role] = session
            session.event("agent_scope", role=role, context_isolated=True, allowed_tools=sorted(permissions))
            return session

        research = make_session("research", self.toolbox, self.backend,
                                self.system_prompt + "\n\n" + RESEARCHER_INSTRUCTIONS, prompt, names)
        self._history = research.inputs

        def finish(status):
            audit = state.get("audit")
            state["phase"] = "done"
            if "audit" in sessions:
                sessions["audit"].finish(status, canonical_json(audit) if audit else "")
            result = research.finish(status, state["answer"], roles=["researcher", "auditor"],
                audit_protocol_version=AUDIT_PROTOCOL_VERSION, citation_check=state.get("citation_check"),
                role_prompt_hash=ROLE_PROMPT_HASH,
                citation_repairs=state.get("citation_repairs", 0), audit_repairs=state.get("audit_repairs", 0),
                model_requests=state["requests"], audit=audit, audit_approved=bool(audit and audit["verdict"] == "approved"),
                evidence_packet_hash=state.get("packet_hash"),
                total_steps=sum(s.steps for s in sessions.values()),
                total_tool_calls=sum(s.tool_calls for s in sessions.values()))
            state["result"] = result
            save()
            return result

        if state["pending_tools"]:
            return finish("reconciliation_required")
        try:
            while state["phase"] in {"research", "cite_repair"}:
                repairing = state["phase"] == "cite_repair"
                turn = research.turn(phase="citation_repair" if repairing else "researcher", use_tools=not repairing)
                if function_calls(turn.output):
                    if repairing:
                        return finish("invalid_citations")
                    state["pending_tools"] = True
                    save()
                    before = len(research.inputs)
                    research.execute(turn)
                    calls = {c.get("call_id") or c.get("id"): c for c in function_calls(turn.output)}
                    for item in research.inputs[before:]:
                        if item.get("type") == "function_call_output":
                            call = calls[item["call_id"]]
                            state["observations"].append({"id": "e" + str(len(state["observations"]) + 1),
                                "tool": call["name"], "arguments": call.get("arguments"),
                                "result": json.loads(item["output"])})
                    state["pending_tools"] = False
                else:
                    if not turn.output_text.strip():
                        return finish("empty_response")
                    state["answer"] = turn.output_text
                    if repairing:
                        from reclab_agent.core.llm import parse_json_response
                        try:
                            corrected = parse_json_response(turn.output_text)
                            original = parse_json_response(state["citation_original"])
                            if (not isinstance(corrected, dict) or
                                    canonical_json({k: v for k, v in corrected.items() if k != "citations"}) !=
                                    canonical_json({k: v for k, v in original.items() if k != "citations"})):
                                raise ValueError("Citation correction changed other claim fields")
                        except (ValueError, TypeError):
                            state["citation_check"] = {"status": "failed", "errors": [
                                {"code": "INVALID_CITATION_CORRECTION", "message": "Correct only citations in the original JSON answer"}]}
                            return finish("invalid_citations")
                    check = check_claim_citations(turn.output_text, state["observations"])
                    state["citation_check"] = check
                    research.event("citation_validation", **check)
                    if check["status"] == "failed":
                        if state.get("citation_repairs", 0) >= 1 or state["requests"] >= self.max_steps - 2:
                            return finish("invalid_citations")
                        state.update(phase="cite_repair", citation_repairs=1)
                        state["citation_original"] = turn.output_text
                        research.inputs.append({"role": "user", "content":
                            CITATION_REPAIR_PROMPT + canonical_json(check)})
                        save()
                        continue
                    packet = {"task": prompt, "research_tools": sorted(names), "observations": state["observations"],
                              "limitations": "Only host-recorded tool observations; no independent filesystem or hidden grader access."}
                    state["packet"] = copy.deepcopy(packet)
                    state["packet_hash"] = content_hash(packet)
                    state["phase"] = "inspect"
                save()

            audit = make_session("audit", AuditToolbox(state["packet"]), self.audit_backend, AUDIT_INSTRUCTIONS,
                AUDIT_INSPECTION_PROMPT + canonical_json(state["packet"]),
                {"read_audit_evidence"})
            while state["phase"] == "inspect":
                turn = audit.turn(phase="audit_inspect")
                if function_calls(turn.output):
                    # Read-only calls need no ambiguous-mutation reconciliation.
                    audit.execute(turn)
                    save()
                    if state["requests"] >= self.max_steps - 1:
                        return finish("audit_incomplete")
                else:
                    if not turn.output_text.strip():
                        return finish("audit_incomplete")
                    state["independent_assessment"] = turn.output_text
                    audit.inputs.append({"role": "user", "content":
                        AUDIT_JUDGMENT_PROMPT +
                        canonical_json({"researcher_claim": state["answer"],
                                        "host_citation_check": state.get("citation_check")})})
                    audit.tools = [copy.deepcopy(AUDIT_SCHEMA)]
                    state["phase"] = "judge"
                    save()
            while state["phase"] == "judge":
                audit.tools = [copy.deepcopy(AUDIT_SCHEMA)]
                turn = audit.turn(phase="audit_judge")
                calls = function_calls(turn.output)
                try:
                    if calls:
                        if len(calls) != 1 or calls[0].get("name") != "submit_audit":
                            return finish("invalid_audit")
                        raw = calls[0].get("arguments", "")
                        value = json.loads(raw) if isinstance(raw, str) else raw
                    else:
                        from reclab_agent.core.llm import parse_json_response
                        value = parse_json_response(turn.output_text)
                    state["audit"] = validate_audit(value, {o["id"] for o in state["observations"]})
                except (ValueError, TypeError, KeyError) as error:
                    code = "INVALID_AUDIT_JSON" if isinstance(error, json.JSONDecodeError) else "INVALID_AUDIT_SCHEMA"
                    audit.event("audit_protocol_error", code=code, message=str(error),
                                error_category="invalid_json" if isinstance(error, json.JSONDecodeError) else "invalid_schema",
                                raw_submission=calls[0].get("arguments") if calls else turn.output_text)
                    if state.get("audit_repairs", 0) >= 1 or state["requests"] >= self.max_steps:
                        return finish("invalid_audit")
                    state["audit_repairs"] = 1
                    for call in calls:
                        call_id = call.get("call_id") or call.get("id")
                        if not call_id:
                            return finish("invalid_audit")
                        audit.inputs.append({"type": "function_call_output", "call_id": call_id,
                            "output": canonical_json({"ok": False, "error": {"code": code, "message": str(error)}})})
                    audit.inputs.append({"role": "user", "content":
                        AUDIT_REPAIR_PROMPT})
                    save()
                    continue
                return finish("completed" if value["verdict"] == "approved" else
                              "audit_rejected" if value["verdict"] == "rejected" else "audit_inconclusive")
        except StepLimitExceeded:
            return finish("step_limit_exceeded")
