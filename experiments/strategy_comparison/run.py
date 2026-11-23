"""Run this experiment protocol without changing the Agent's product behavior."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import os
from reclab_agent.domain.contracts import Budget, canonical_json
from reclab_agent.infrastructure.files import atomic_write
from reclab_agent.domain.configuration import load_task_configuration, read_json_strict
from reclab_agent.reporting import format_user_summary
from .controller import run_comparison
from .settings import ComparisonConfiguration, with_comparison
from .runner import bind_experiment_tools


def prepare(protocol_path):
    path = Path(os.path.abspath(protocol_path))
    protocol = read_json_strict(path, max_bytes=65_536)
    required = {"task_config", "env_file", "output_directory", "strategies", "candidates_per_strategy",
                "model_calls_per_strategy", "patches_per_strategy", "max_plan_steps", "max_reflections",
                "max_job_seconds", "seed"}
    if not isinstance(protocol, dict) or protocol.keys() != required:
        raise ValueError("experiment protocol has unexpected or missing fields")
    for key in ("model_calls_per_strategy", "patches_per_strategy", "max_job_seconds"):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError("experiment limits must be positive integers")
    seed = protocol["seed"]
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("experiment seed must be a nonnegative 31-bit integer")
    settings = ComparisonConfiguration(tuple(protocol["strategies"]), protocol["candidates_per_strategy"],
                                       protocol["max_plan_steps"], protocol["max_reflections"])
    base = load_task_configuration(path.parent / protocol["task_config"])
    modes = len(settings.strategies)
    jobs = 1 + modes * settings.candidates_per_strategy
    budget = Budget(max_model_turns=1 + modes * protocol["model_calls_per_strategy"],
                    max_patches=modes * protocol["patches_per_strategy"], max_training_jobs=jobs,
                    max_job_seconds=protocol["max_job_seconds"],
                    max_total_job_seconds=jobs * protocol["max_job_seconds"],
                    max_cost_cny=base.task.budget.max_cost_cny)
    settings.validate_budget(budget)
    config = with_comparison(base, settings,
        task=replace(base.task, question="Fairly compare ReAct, Plan and Execute, and Reflection under identical candidate budgets.",
                     budget=budget),
        candidate_ids=tuple(f"candidate-{index}" for index in range(1, settings.candidates_per_strategy + 1)),
        seeds=(seed,), output_directory=Path(os.path.abspath(path.parent / protocol["output_directory"])))
    return config, Path(os.path.abspath(path.parent / protocol["env_file"])), protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--run", action="store_true", help="execute this experiment; default only prints the protocol")
    parser.add_argument("--windows-ssh", action="store_true", help="use Windows OpenSSH when Python runs under WSL")
    parser.add_argument("--json", action="store_true", help="print the final machine-readable result instead of the user summary")
    args = parser.parse_args(argv)
    config, environment, protocol = prepare(args.protocol)
    print(canonical_json({"strategies": list(config.comparison.strategies),
        "candidates_per_strategy": config.comparison.candidates_per_strategy,
        "training_attempt_limit": config.task.budget.max_training_jobs,
        "model_call_limit": config.task.budget.max_model_turns,
        "reserved_experiment_seconds": config.task.budget.max_total_job_seconds,
        "seed": config.seeds[0], "output_directory": str(config.output_directory)}), flush=True)
    if not args.run:
        return 0
    config.output_directory.mkdir(parents=True, exist_ok=True)
    atomic_write(config.output_directory / "protocol.json", canonical_json(protocol).encode("utf-8"))
    report = run_comparison(config, approve_plan=True, env_file=environment, windows_ssh=args.windows_ssh,
                            binder=bind_experiment_tools,
                            progress=lambda message: print(message, flush=True))
    if args.json:
        print(canonical_json({"status": report["status"], "report_path": report["report_path"],
                              "user_report_path": report.get("user_report_path"),
                              "user_report_markdown_path": report.get("user_report_markdown_path"),
                              "comparison": report["comparison"], "winning_strategies": report["winning_strategies"]}), flush=True)
    else:
        print(format_user_summary(report), flush=True)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
