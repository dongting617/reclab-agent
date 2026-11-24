"""Local evaluation commands; model calls are opt-in, GPU jobs are never started."""
import argparse
import os
from pathlib import Path

from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.infrastructure.files import atomic_write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    retrieval = sub.add_parser("retrieval", help="evaluate annotated paragraphs with BM25 or CPU text models")
    retrieval.add_argument("--manifest", type=Path, default=Path("tasks/retrieval.json"))
    retrieval.add_argument("--modes", nargs="+", choices=("bm25", "dense", "hybrid", "reranked"), default=["bm25"])
    retrieval.add_argument("--output", type=Path, default=Path("reports/local-retrieval.json"))
    retrieval.add_argument("--k", type=int, default=5)
    retrieval.add_argument("--embedding-model", default="BAAI/bge-small-zh-v1.5")
    retrieval.add_argument("--reranking-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    retrieval.add_argument("--embedding-revision")
    retrieval.add_argument("--reranking-revision")
    retrieval.add_argument("--download-models", action="store_true", help="allow one-time CPU model downloads")
    diagnostic = sub.add_parser("diagnostics", help="evaluate frozen synthetic scenarios; reference mode makes no API calls")
    diagnostic.add_argument("--manifest", type=Path, default=Path("tasks/diagnostic-v2/diagnostic-dev.json"))
    diagnostic.add_argument("--output", type=Path, default=Path("reports/local-diagnostics.json"))
    diagnostic.add_argument("--mode", choices=("reference", "react", "plan_execute", "reflection", "team"), default="reference")
    diagnostic.add_argument("--model", default=os.environ.get("RECLAB_MODEL"))
    diagnostic.add_argument("--max-steps", type=int, default=8)
    diagnostic.add_argument("--max-requests", type=int, default=120)
    diagnostic.add_argument("--prices", type=Path)
    diagnostic.add_argument("--env-file", type=Path)
    coding = sub.add_parser("coding", help="compare actual code edits using frozen CPU Docker checks")
    coding.add_argument("--manifest", type=Path, default=Path("tasks/coding.json"))
    coding.add_argument("--output", type=Path, default=Path("reports/local-coding.json"))
    coding.add_argument("--image", required=True, help="immutable locally cached sha256 image ID")
    coding.add_argument("--mode", choices=("reference", "react", "team"), default="reference")
    coding.add_argument("--model")
    coding.add_argument("--env-file", type=Path)
    coding.add_argument("--repetitions", type=int, choices=(1,2), default=2)
    coding.add_argument("--max-steps", type=int, default=8)
    coding.add_argument("--max-requests", type=int, default=64)
    costs = sub.add_parser("costs", help="summarize a saved trace using explicitly supplied token prices")
    costs.add_argument("--trace", required=True, type=Path)
    costs.add_argument("--prices", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "retrieval":
            from reclab_agent.evaluation.retrieval import evaluate_retrieval
            from reclab_agent.integrations.text_models import TextModels
            models = None
            if any(mode != "bm25" for mode in args.modes):
                models = TextModels(args.embedding_model, args.reranking_model,
                                    args.embedding_revision, args.reranking_revision, not args.download_models)
            result = evaluate_retrieval(args.manifest, modes=args.modes, models=models, k=args.k)
            atomic_write(args.output, canonical_json(result).encode("utf-8"))
            summary = {"output": str(args.output), "manifest_hash": result["manifest_hash"],
                       "runs": [{k: v for k, v in row.items() if k != "results"} for row in result["runs"]]}
        elif args.command in {"diagnostics", "coding"}:
            from reclab_agent.core import OpenAIResponsesBackend
            from reclab_agent.evaluation.diagnostics import run_diagnostic_suite
            if args.mode != "reference" and args.env_file:
                from types import SimpleNamespace
                from reclab_agent.application.environment import load_agent_environment
                load_agent_environment(args.env_file, SimpleNamespace(model_env="RECLAB_MODEL",
                    api_key_env="OPENAI_API_KEY", tavily_key_env="TAVILY_API_KEY"))
            args.model = args.model or os.environ.get("RECLAB_MODEL")
            if args.mode != "reference" and not args.model:
                raise ValueError("--model or RECLAB_MODEL is required for actual agent evaluation")
            backend = OpenAIResponsesBackend(args.model) if args.mode != "reference" else None
            if backend is not None:
                # Count each HTTP attempt at the evaluator boundary; SDK retries
                # would otherwise consume unrecorded requests outside its quota.
                backend.client = backend.client.with_options(max_retries=0, timeout=40)
            if args.command == "coding":
                from reclab_agent.evaluation.coding import run_coding_suite
                from reclab_agent.execution.coding import CodingContainer
                result = run_coding_suite(args.manifest, args.output, executor=CodingContainer(args.image),
                    mode=args.mode, backend=backend, repetitions=args.repetitions,
                    max_steps=args.max_steps, max_requests=args.max_requests)
                summary = {k:result[k] for k in ("kind","status","trials","passed","pass_rate","model_requests","resource_usage")}
            else:
                result = run_diagnostic_suite(args.manifest, args.output, mode=args.mode, backend=backend,
                                             max_steps=args.max_steps, max_requests=args.max_requests,
                                             prices=read_json_strict(args.prices) if args.prices else None)
                summary = {k: result[k] for k in ("kind", "status", "tasks", "passed", "pass_rate", "model_requests", "resource_usage")}
            summary["output"] = str(args.output)
        else:
            from reclab_agent.evaluation.costs import summarize_usage
            from reclab_agent.evaluation.traces import load_trace
            summary = summarize_usage(load_trace(args.trace), read_json_strict(args.prices) if args.prices else None)
        print(canonical_json({"ok": True, "result": summary}))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(canonical_json({"ok": False, "error": type(error).__name__, "detail": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
