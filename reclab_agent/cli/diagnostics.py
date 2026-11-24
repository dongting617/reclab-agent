from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from reclab_agent.evaluation.benchmark import load_tasks, run_reference_suite
from reclab_agent.evaluation.traces import replay_trace
from reclab_agent.agents import create_agent
from reclab_agent.agents.factory import AGENT_TYPES
from reclab_agent.core.llm import OpenAIResponsesBackend
from reclab_agent.tools.evidence import EvidenceToolbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RecLab evidence tasks or an Agent strategy")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="RecLab Agent repository root")
    parser.add_argument("--tasks", type=Path, help="Task manifest; defaults to tasks/dev.json")
    parser.add_argument("--list-tools", action="store_true", help="Print tool schemas instead of running tasks")
    parser.add_argument("--ask", help="Run a natural-language evidence task with an Agent")
    parser.add_argument("--agent", choices=tuple(AGENT_TYPES), default="react",
                        help="Execution strategy for --ask; default is ReAct")
    parser.add_argument("--max-plan-steps", type=int, default=5, help="Maximum steps in a generated plan")
    parser.add_argument("--max-reflections", type=int, default=1, help="Maximum critique/revision cycles")
    parser.add_argument("--allowed-tools", nargs="+",
                        help="Restrict --ask to these task-level tools")
    parser.add_argument("--replay", type=Path, help="Replay and evaluate a saved JSONL trace")
    parser.add_argument("--expected-tools", nargs="+", default=[],
                        help="Expected tool names for --replay evaluation")
    parser.add_argument("--model", default=os.environ.get("RECLAB_MODEL"),
                        help="OpenAI model ID; may also be set with RECLAB_MODEL")
    parser.add_argument("--trace", type=Path, default=Path("traces/latest.jsonl"),
                        help="JSONL trace path for --ask")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum model turns for --ask")
    args = parser.parse_args()

    root = Path(os.path.abspath(os.fspath(args.root)))
    toolbox = EvidenceToolbox(root)
    if args.ask and args.replay:
        parser.error("--ask and --replay are mutually exclusive")
    if args.ask:
        if not args.model:
            parser.error("--ask requires --model or the RECLAB_MODEL environment variable")
        options = {}
        if args.agent == "plan_execute":
            options["max_plan_steps"] = args.max_plan_steps
        elif args.agent in {"reflection", "reviewer"}:
            options["max_reflections"] = args.max_reflections
        agent = create_agent(args.agent, toolbox, OpenAIResponsesBackend(args.model),
                             max_steps=args.max_steps, trace_path=args.trace, **options)
        payload = agent.run(args.ask, allowed_tools=set(args.allowed_tools)
                            if args.allowed_tools else None)
    elif args.replay:
        payload = replay_trace(args.replay, args.expected_tools)
    elif args.list_tools:
        payload = [{"name": spec.name, "description": spec.description,
                    "input_schema": spec.input_schema} for spec in toolbox.specs]
    else:
        task_path = args.tasks or root / "tasks" / "dev.json"
        payload = run_reference_suite(toolbox, load_tasks(task_path))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
