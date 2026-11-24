"""Bounded live ReAct/research-audit regression, using existing frozen suites."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from reclab_agent.application.environment import load_agent_environment
from reclab_agent.core import OpenAIResponsesBackend
from reclab_agent.core.llm import parse_json_response
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.evaluation.diagnostics import (
    EvaluationRequestLimit, grade_answer, load_diagnostic_tasks, run_diagnostic_suite, validate_answer,
)
from reclab_agent.execution.locking import job_lock
from reclab_agent.infrastructure.files import atomic_write
from reclab_agent.tools.audit import AUDIT_PROTOCOL_VERSION
from reclab_agent.agents.research_audit_prompts import ROLE_PROMPT_HASH


def write(path, value):
    atomic_write(path, canonical_json(value).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".reclab/agent.env"))
    parser.add_argument("--name", default="research-audit-live-20261124")
    parser.add_argument("--max-requests", type=int, default=480)
    parser.add_argument("--minutes", type=int, default=40)
    parser.add_argument("--modes", nargs="+", choices=("react", "research_audit"), default=["react", "research_audit"])
    parser.add_argument("--splits", nargs="+", choices=("dev", "frozen"), default=["dev", "frozen"])
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if not args.name.replace("-", "").isalnum() or args.max_requests < 1 or args.minutes < 1:
        parser.error("invalid name or limits")
    root = Path.cwd()
    sources = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path("reclab_agent").rglob("*.py"))}
    sources["scripts/evaluate_research_audit.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifests = {split: load_diagnostic_tasks(Path("tasks/diagnostic-v2") / ("diagnostic-"+split+".json"))
                 for split in args.splits}
    load_agent_environment(args.env_file, SimpleNamespace(model_env="RECLAB_MODEL", api_key_env="OPENAI_API_KEY",
                                                         tavily_key_env="TAVILY_API_KEY"))
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=40)
    backend = OpenAIResponsesBackend(os.environ["RECLAB_MODEL"], client=client)
    # Same output ceiling for both modes; incomplete output remains a real failure.
    original_create = client.responses.create
    def create(**kwargs):
        return original_create(max_output_tokens=2048, **kwargs)
    client.responses.create = create
    protocol = {"model": backend.model, "source_root": str(root.resolve()), "audit_protocol_version": AUDIT_PROTOCOL_VERSION, "role_prompt_hash": ROLE_PROMPT_HASH, "modes": args.modes,
        "splits": args.splits, "tasks_per_split": 30, "max_steps": 8,
        "batch_request_cap": 240, "global_request_cap": args.max_requests, "minutes": args.minutes,
        "sdk_retries": 0, "agent_retries": 0, "timeout_seconds": 40, "max_output_tokens": 2048,
        "source_hash": content_hash(sources), "source_files": sources,
        "manifest_hashes": {s: m[1] for s, m in manifests.items()},
        "gpu_jobs": 0, "coding": {"status": "not_run", "reason": "Not included in this diagnostic evaluation"},
        "limitations": ["Previously inspected frozen cases are regression cases, not an unseen generalization test.",
                        "Synthetic diagnostic decisions do not establish end-to-end research or coding performance.",
                        "Scene variants are correlated; identical request caps do not imply equal actual cost.",
                        "Fixed execution order and provider caching affect observed latency."]}
    path = args.output_dir / (args.name + ".json")
    identity = content_hash(protocol)
    state = {"identity": identity, "protocol": protocol, "requests": 0,
             "deadline": time.time()+args.minutes*60, "batches": [], "status": "running"}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise ValueError("run configuration changed; use a new name")
    write(path, state)

    class SharedBackend:
        model = backend.model
        def respond(self, **kwargs):
            if state["requests"] >= args.max_requests or time.time() >= state["deadline"]:
                raise EvaluationRequestLimit("live evaluation request/time allowance exhausted")
            state["requests"] += 1
            write(path, state)
            return backend.respond(**kwargs)

    for split in protocol["splits"]:
        for mode in protocol["modes"]:
            output = args.output_dir / (args.name + "-" + split + "-" + mode + ".json")
            print(json.dumps({"event": "batch_started", "split": split, "mode": mode,
                              "reserved_requests": state["requests"]}), flush=True)
            result = run_diagnostic_suite(Path("tasks/diagnostic-v2") / ("diagnostic-"+split+".json"),
                output, mode=mode, backend=SharedBackend(), max_steps=8, max_requests=240)
            tasks = {r["id"]: r for r in manifests[split][0]["tasks"]}
            audit_rows = []
            for row in result["results"]:
                if mode != "research_audit" or not row.get("trace"):
                    continue
                checkpoint = Path(row["trace"] + ".audit.json")
                if not checkpoint.exists():
                    continue
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                valid, graded = False, {"decision_correct": False, "passed": False}
                try:
                    answer = parse_json_response(saved.get("answer", ""))
                    validate_answer(answer)
                    valid = True
                    graded = grade_answer(answer, tasks[row["id"]])
                except (ValueError, TypeError, KeyError):
                    pass
                audit_rows.append({"id": row["id"], "agent_status": row.get("agent_status"),
                    "answer_valid": valid, "research_decision_correct": graded["decision_correct"],
                    "research_answer_passed": graded["passed"], "audit": saved.get("audit"),
                    "phase": saved.get("phase"), "model_requests": saved.get("requests")})
            batch = {"split": split, "mode": mode, "report": output.as_posix(),
                "tasks": result["tasks"], "passed": result["passed"], "pass_rate": result["pass_rate"],
                "status": result["status"], "model_requests": result["model_requests"],
                "resource_usage": result["resource_usage"], "scene_groups": result["scene_groups"],
                "failures": dict(Counter(r.get("failure", {}).get("code", r.get("error_type", "unknown"))
                                         for r in result["results"] if not r["passed"])),
                "audit_results": audit_rows}
            state["batches"] = [b for b in state["batches"] if (b["split"], b["mode"]) != (split, mode)] + [batch]
            write(path, state)
            print(json.dumps({k: v for k, v in batch.items() if k not in ("resource_usage", "audit_results", "scene_groups")}), flush=True)
    state["status"] = "completed" if all(b["status"] == "completed" for b in state["batches"]) else "partial"
    state["completed_at"] = time.time()
    write(path, state)
    print(json.dumps({"event": "evaluation_finished", "status": state["status"], "requests": state["requests"]}), flush=True)


if __name__ == "__main__":
    with job_lock(Path(".reclab"), "research-audit-evaluation") as acquired:
        if not acquired:
            raise SystemExit("Another evaluation owns the run lock")
        main()
