"""Frozen synthetic diagnostic tasks evaluated without exposing labels to agents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time

from reclab_agent.agents import create_agent
from reclab_agent.agents.research_audit_prompts import ROLE_PROMPT_HASH
from reclab_agent.core.llm import parse_json_response
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.contracts import canonical_json, content_hash, identifier
from reclab_agent.evaluation.costs import summarize_usage
from reclab_agent.infrastructure.files import atomic_write
from reclab_agent.tools import EvidenceToolbox, json_pointer
from reclab_agent.tools.audit import AUDIT_PROTOCOL_VERSION


PROMPT = """You are a recommendation experiment analyst.
Read the current case evidence and decide whether to retain, reject, mark inconclusive, or stop.

1. Read the evidence
Call read_json_values with these arguments:
{
  "path": "evidence/case.json",
  "pointers": [""]
}
Read the complete evidence in one call. Do not inspect metadata or repeat tool calls
when the returned evidence already answers the question.

2. Decision rules
Apply these rules in order. Stop at the first matching rule:
- stop: No training slots remain, or the source duplicates an evaluated candidate.
- inconclusive: The stop rule does not apply, and at least one condition holds:
  - Stopping is unconfirmed.
  - Baseline and candidate differ in dataset, evaluation protocol, or random seed.
  - Training did not complete.
  - Independent metrics are missing.
  - Required process diagnostics did not pass.
- retain: None of the conditions above apply, candidate ndcg@20 is higher than baseline,
  and candidate recall@20 is no lower than baseline.
- reject: None of the stop or inconclusive conditions apply, but the candidate does
  not satisfy the retain criteria.

3. Citation requirements
- path must be evidence/case.json.
- Each pointer must resolve to an observed field directly relevant to the decision.
- Field paths start with /, such as /remaining_training_jobs or /metrics/baseline/ndcg@20.
- The empty string is the root pointer. / refers to an empty object key, not the root.
- Escape ~ as ~0 and / as ~1 only inside a field name. Keep path separators unchanged.
- The @ character in ndcg@20 and recall@20 does not need escaping.
- Distinguish observations from inferred mechanisms. Do not invent metrics, fields, or results.

4. Output requirements
Return exactly one valid JSON object. Do not add introductory text, Markdown fences,
or any text outside the object.
Required fields:
- decision: One of retain, reject, inconclusive, or stop.
- reason: A nonempty string explaining why the current evidence supports the decision.
- citations: A nonempty array of objects containing path and pointer. Cite multiple
  relevant fields when needed.

5. Output example
This example assumes the evidence reports zero remaining training slots. It illustrates
the format only. Fill in the actual values and decision from the current case;
do not copy the example conclusion.
{
  "decision": "stop",
  "reason": "The evidence reports remaining_training_jobs as 0, so no training slots remain and further experiments must stop.",
  "citations": [
    {
      "path": "evidence/case.json",
      "pointer": "/remaining_training_jobs"
    }
  ]
}"""

FAILURE_POLICY = "separate_decision_and_citation_validation_v3"


class EvaluationRequestLimit(ValueError):
    request_not_sent = True


def validate_answer(answer):
    """Validate the response shape separately from correctness and citations."""
    if (not isinstance(answer, dict) or answer.get("decision") not in
            ("retain", "reject", "inconclusive", "stop") or
            not isinstance(answer.get("reason"), str) or not answer["reason"].strip() or
            not isinstance(answer.get("citations"), list) or not answer["citations"]):
        raise ValueError("invalid diagnostic answer schema")
    if any(not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or
           not isinstance(ref.get("pointer"), str) for ref in answer["citations"]):
        raise ValueError("invalid diagnostic citation schema")


def reference_decision(evidence):
    if evidence["remaining_training_jobs"] <= 0 or evidence["duplicate_source"]:
        return "stop"
    if (not evidence["stop_confirmed"] or evidence["scope"]["baseline"] != evidence["scope"]["candidate"]
            or evidence["status"] != "completed" or not evidence["metrics"] or not evidence["required_process_pass"]):
        return "inconclusive"
    before, after = evidence["metrics"]["baseline"], evidence["metrics"]["candidate"]
    return "retain" if after["ndcg@20"] > before["ndcg@20"] and after["recall@20"] >= before["recall@20"] else "reject"


def load_diagnostic_tasks(path):
    path = Path(path).absolute()
    raw = path.read_bytes()
    payload = read_json_strict(path, max_bytes=4_000_000)
    freeze = read_json_strict(path.parent / "local-evaluation-freeze.json")
    if freeze["files"].get(path.name) != hashlib.sha256(raw).hexdigest():
        raise ValueError("diagnostic manifest differs from its frozen version")
    tasks = payload["tasks"]
    if payload.get("schema_version") != 1 or not tasks or len({r["id"] for r in tasks}) != len(tasks):
        raise ValueError("invalid diagnostic manifest")
    for row in tasks:
        identifier(row["id"])
        if row["expected_decision"] not in {"retain", "reject", "inconclusive", "stop"}:
            raise ValueError("invalid expected decision")
        json_pointer(row["evidence"], row["required_pointer"])
    return payload, hashlib.sha256(raw).hexdigest()


def grade_answer(answer, task):
    decision_ok = isinstance(answer, dict) and answer.get("decision") == task["expected_decision"]
    try:
        validate_answer(answer)
    except (ValueError, TypeError):
        return {"decision_correct": decision_ok, "citation_located": False, "citations_valid": False,
                "citation_errors": [{"code": "INVALID_ANSWER_SCHEMA"}], "passed": False}
    located, errors = [], []
    for index, ref in enumerate(answer["citations"]):
        try:
            if ref["path"] != "evidence/case.json":
                raise ValueError("citation outside this case")
            pointer = ref["pointer"]
            json_pointer(task["evidence"], pointer)
            required = task["required_pointer"]
            located.append(pointer == required or pointer.startswith(required + "/"))
        except (ValueError, KeyError, TypeError) as error:
            errors.append({"index": index, "code": "INVALID_CITATION", "message": str(error)})
    return {"decision_correct": decision_ok, "citation_located": any(located), "citations_valid": not errors,
            "citation_errors": errors, "passed": decision_ok and any(located) and not errors}


def run_diagnostic_suite(manifest_path, output, *, mode="reference", backend=None,
                         max_steps=8, max_requests=120, prices=None):
    if type(max_steps) is not int or max_steps < 1 or type(max_requests) is not int or max_requests < 1:
        raise ValueError("model limits must be positive integers")
    manifest, digest = load_diagnostic_tasks(manifest_path)
    if mode != "reference" and backend is None:
        raise ValueError("agent evaluation requires an explicit model backend")
    output = Path(output).absolute()
    trace_root = output.parent / (output.stem + "-traces")
    agent_protocol = AUDIT_PROTOCOL_VERSION if mode == "research_audit" else None
    role_prompt_hash = ROLE_PROMPT_HASH if mode == "research_audit" else None
    identity = content_hash({"manifest": digest, "prompt": content_hash(PROMPT), "role_prompt_hash": role_prompt_hash, "response_format": "json_or_single_fence_v1", "success_policy": "completed_agent_and_valid_answer_v1", "failure_policy": FAILURE_POLICY, "agent_protocol": agent_protocol, "mode": mode, "model": getattr(backend, "model", None),
                             "max_steps": max_steps, "max_requests": max_requests, "prices": prices})
    report = {"kind": "reference_contract" if mode == "reference" else "synthetic_agent_diagnostics",
              "identity": identity, "manifest_hash": digest, "prompt_hash": content_hash(PROMPT), "role_prompt_hash": role_prompt_hash, "failure_policy": FAILURE_POLICY, "agent_protocol": agent_protocol, "split": manifest["split"], "mode": mode,
              "model": getattr(backend, "model", None), "model_requests": 0, "results": [],
              "scope": manifest["origin"], "gpu_jobs": 0, "status": "running",
              "limitations": ["Synthetic task success is not real coding or recommendation quality.",
                             "Citation location checks do not prove every claim is factually supported.",
                             "Variants in a scene group are correlated; do not treat them as independent trials."]}
    if output.exists():
        report = read_json_strict(output, max_bytes=8_000_000)
        if report["identity"] != identity:
            raise ValueError("output is bound to another evaluation; choose a new output")
    def save():
        atomic_write(output, canonical_json(report).encode("utf-8"))
    class LimitedBackend:
        model = getattr(backend, "model", None)
        def respond(self, **kwargs):
            if report["model_requests"] >= max_requests:
                raise EvaluationRequestLimit("shared evaluation model request limit exhausted")
            report["model_requests"] += 1
            save()
            return backend.respond(**kwargs)
    limited = LimitedBackend()
    completed = {r["id"] for r in report["results"]}
    provider_failures = 0
    for task in manifest["tasks"]:
        if task["id"] in completed:
            continue
        row = {"id": task["id"], "scene_group": task["scene_group"], "status": "failed",
               "passed": False, "decision_correct": False, "citation_located": False,
               "citations_valid": False, "citation_errors": []}
        if report["model_requests"] >= max_requests or provider_failures >= 3:
            row.update(status="not_run", error_type="RequestLimit" if report["model_requests"] >= max_requests else "RepeatedProviderFailure")
            report["results"].append(row)
            save()
            continue
        with tempfile.TemporaryDirectory(prefix="reclab-evidence-") as directory:
            root = Path(directory)
            atomic_write(root / "evidence/case.json", canonical_json(task["evidence"]).encode("utf-8"))
            tools = EvidenceToolbox(root)
            started = time.perf_counter()
            phase = "agent"
            agent = None
            try:
                if mode == "reference":
                    measured = tools.invoke("read_json_values", {"path": "evidence/case.json", "pointers": [""]})
                    answer = {"decision": reference_decision(measured["result"]["values"][""]),
                              "reason": "Deterministic reference policy applied to recorded evidence.",
                              "citations": [{"path": "evidence/case.json", "pointer": task["required_pointer"]}]}
                else:
                    trace = trace_root / (task["id"] + ".jsonl")
                    row["trace"] = str(trace)
                    # Each task is a separate immutable run; labels never enter this toolbox.
                    agent = create_agent(mode, tools, limited, max_steps=max_steps, model_retries=0, trace_path=trace)
                    result = agent.run(PROMPT, task_id=task["id"],
                                       allowed_tools={"read_json_values", "compare_numeric_values", "inspect_artifact"})
                    row["agent_status"] = result["status"]
                    row["elapsed_ms"] = result["elapsed_ms"]
                    row["raw_answer"] = result["answer"]
                    if result["status"] != "completed":
                        row["failure"] = {"category": "agent_termination", "code": result["status"]}
                        if not result["answer"].strip():
                            raise ValueError("agent terminated before producing an answer")
                    phase = "json"
                    answer = parse_json_response(result["answer"])
                if isinstance(answer, dict):
                    row["decision_correct"] = answer.get("decision") == task["expected_decision"]
                phase = "schema"
                validate_answer(answer)
                phase = "grading"
                graded = grade_answer(answer, task)
                finished = mode == "reference" or row.get("agent_status") == "completed"
                graded["passed"] = graded["passed"] and finished
                row.update(graded, answer=answer, status="completed" if finished else "failed")
                if not graded["passed"]:
                    row.setdefault("failure", {"category": "grading", "code":
                        "INVALID_CITATION" if not graded["citations_valid"] else
                        "INCORRECT_DECISION" if not graded["decision_correct"] else "CITATION_NOT_LOCATED"})
                provider_failures = 0
            except Exception as error:
                row["error_type"] = type(error).__name__
                provider_error = type(error).__module__.split(".")[0] == "openai"
                if provider_error:
                    provider_failures += 1
                else:
                    provider_failures = 0
                if isinstance(error, EvaluationRequestLimit):
                    row["status"] = "budget_exceeded"
                    failure = {"category": "budget", "code": "REQUEST_LIMIT"}
                elif provider_error:
                    failure = {"category": "provider", "code":
                        "PROVIDER_TIMEOUT" if "Timeout" in type(error).__name__ else "PROVIDER_ERROR"}
                elif phase in {"json", "schema"}:
                    failure = {"category": "response_format", "code":
                        "INVALID_JSON_RESPONSE" if phase == "json" else "INVALID_ANSWER_SCHEMA"}
                else:
                    failure = {"category": "runtime", "code": type(error).__name__}
                row.setdefault("failure", failure)
                if "failure" in row and phase == "json":
                    row["failure"]["parse_error_type"] = type(error).__name__
                checkpoint = getattr(agent, "checkpoint_path", None)
                if checkpoint and Path(checkpoint).exists():
                    try:
                        saved = read_json_strict(checkpoint, max_bytes=20_000_000)
                        row["failure"]["role_phase"] = saved.get("phase")
                    except (ValueError, OSError):
                        pass  # Optional metadata must never replace the original failure.
            row.setdefault("elapsed_ms", round((time.perf_counter() - started) * 1000, 3))
        report["results"].append(row)
        save()
    rows = report["results"]
    report["passed"] = sum(r["passed"] for r in rows)
    report["tasks"] = len(manifest["tasks"])
    report["pass_rate"] = report["passed"]/report["tasks"]
    report["scene_groups"] = {group: {"tasks": sum(r["scene_group"] == group for r in rows),
                                     "passed": sum(r["passed"] for r in rows if r["scene_group"] == group)}
                              for group in sorted({r["scene_group"] for r in rows})}
    events = []
    for path in trace_root.glob("*.jsonl"):
        events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    report["resource_usage"] = summarize_usage(events, prices)
    unreported = report["model_requests"] - report["resource_usage"]["model_requests"]
    if unreported > 0:
        events.extend({"event": "model_turn", "usage": None} for _ in range(unreported))
        report["resource_usage"] = summarize_usage(events, prices)
    report["status"] = "completed" if all(r["status"] == "completed" for r in rows) else "partial"
    save()
    return report
