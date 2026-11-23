"""Inspect inputs, launch a configured optimization Agent, or run the fixture demo."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import reclab_agent.examples.demo as demo
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.execution.worker import CapabilityError, DockerError


def _error(error: BaseException) -> dict:
    return {"ok": False, "error": type(error).__name__, "detail": str(error), "host_fallback": False}


def _child(root: str, image: str, operation: str, timeout: int) -> dict:
    # This invokes trusted controller code, never src/train.py or src/predict.py.
    completed = subprocess.run([sys.executable, "-m", "reclab_agent.cli.research", operation,
                                "--root", root, "--image", image],
                               shell=False, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        raise DockerError("trusted demo controller returned no valid JSON result") from error
    if not isinstance(result, dict) or completed.returncode not in (0, 2):
        raise DockerError("trusted demo controller failed unexpectedly")
    return result


def run_supervised(root: str, image: str) -> dict:
    """Hard 120 s controller wait, then a separately capped 30 s stop confirmation."""
    root = str(demo.load(root)[0])
    with demo.run_lock(root):
        demo.policy_for(root, image)  # Reject mutable tags before binding a run.
        previous = demo.status(root)["run"]
        if previous and previous.get("image") != image:
            raise ValueError("a demo root is bound to one image; prepare another root to change it")
        try:
            result = _child(root, image, "_execute", demo.RUN_SECONDS)
        except subprocess.TimeoutExpired:
            result = _error(TimeoutError("demo controller exceeded its 120-second wall-clock limit"))
        except KeyboardInterrupt:
            result = _error(KeyboardInterrupt("demo interrupted; requesting cancellation"))
        except (DockerError, OSError) as error:
            result = _error(error)
        if result.get("ok") is True:
            return result
        try:
            stopped = _child(root, image, "_cancel", demo.STOP_SECONDS)
            stop = stopped.get("result", {"stop_confirmed": False, "detail": stopped.get("detail", "cancellation failed")})
        except (subprocess.TimeoutExpired, OSError, DockerError, KeyboardInterrupt) as error:
            stop = {"stop_confirmed": False, "detail": f"stop confirmation interrupted: {type(error).__name__}"}
        result["stop"] = stop
        result["operator_action"] = (
            "Keep the independent watchdog running to catch late starts; exit is confirmed only at the recorded observation."
            if stop.get("continued_watchdog_required") and stop.get("stop_confirmed") else
            None if stop.get("stop_confirmed") else
            "Keep an independent watchdog running and inspect the Docker jobs; stopping is unconfirmed.")
        demo._write(Path(root), "last-error.json", result)
        demo.refresh_failed_run(root, result, stop)
        return result


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Private controller entrypoints are fixed operations, never arbitrary argv/code.
    internal = bool(argv and argv[0] in {"_execute", "_cancel"})
    parser = argparse.ArgumentParser(description=__doc__)
    if internal:
        parser.add_argument("operation", choices=("_execute", "_cancel"))
        parser.add_argument("--root", required=True)
        parser.add_argument("--image", required=True)
    else:
        commands = parser.add_subparsers(dest="command", required=True)
        check = commands.add_parser("check", help="validate a task configuration and its inputs without running experiments")
        check.add_argument("--config", type=Path, required=True, help="task configuration JSON; paths are relative to this file")
        check.set_defaults(operation="check")
        for name in ("plan", "api-check", "run", "resume"):
            command = commands.add_parser(name, help={
                "plan": "show the baseline/candidate scope without API requests or jobs",
                "api-check": "test Responses API tool calls without starting GPU jobs",
                "run": "run the real baseline/candidate optimization Agent",
                "resume": "resume an unfinished research session without resetting budgets",
            }[name])
            command.add_argument("--config", type=Path, required=True)
            command.set_defaults(operation=name)
            if name != "plan":
                command.add_argument("--env-file", type=Path, help="load named API settings as data, without executing shell")
            if name in {"run", "resume"}:
                if name == "resume":
                    command.add_argument("--session", type=Path, required=True)
                command.add_argument("--json", action="store_true",
                                     help="print the structured result instead of the user summary")
                command.add_argument("--approve-plan", action="store_true",
                                     help="explicitly approve the configured scope shown by the plan command")
                command.add_argument("--windows-ssh", action="store_true",
                                     help="use Windows OpenSSH from the WSL control process")
        project = commands.add_parser("project", help="manage project resources, ledger and knowledge").add_subparsers(dest="project_operation", required=True)
        for operation in ("init", "experiments", "runs", "search", "reindex", "import", "note"):
            command = project.add_parser(operation)
            command.add_argument("--config", type=Path, required=True)
            command.set_defaults(operation="project")
            if operation == "experiments":
                command.add_argument("--experiment-id")
            if operation == "search":
                command.add_argument("--query", required=True)
            if operation in {"import", "note"}:
                command.add_argument("--input", type=Path, required=True)
            if operation == "note":
                command.add_argument("--note-id", required=True)
                command.add_argument("--replaces")
        report = commands.add_parser("report", help="render a saved experiment report without API or GPU calls")
        report.add_argument("--input", type=Path, required=True, help="saved experiment report JSON")
        report.add_argument("--output-directory", type=Path, help="write report.html and report.md here; defaults beside input")
        report.add_argument("--json", action="store_true", help="print generated report paths as JSON")
        report.set_defaults(operation="report")
        commands = commands.add_parser("demo", help="audited fixture, not an LLM research agent").add_subparsers(dest="operation", required=True)
        for name in ("prepare", "status", "doctor", "run"):
            command = commands.add_parser(name)
            command.add_argument("--root", required=True, help="new directory for prepare; prepared root otherwise")
            if name in {"doctor", "run"}:
                command.add_argument("--image", required=True, help="pre-pulled administrator-approved repository@sha256:digest")
    args = parser.parse_args(argv)
    try:
        if args.operation == "project":
            from reclab_agent.cli.projects import project_command
            print(canonical_json({"ok": True, "result": project_command(args)}))
            return 0
        if args.operation == "check":
            from reclab_agent.domain.configuration import load_task_configuration
            from reclab_agent.application.inspection import inspect_task
            value = inspect_task(load_task_configuration(args.config))
            result = {"ok": value["inputs_valid"], "result": value}
            print(canonical_json(result))
            return 0 if result["ok"] else 2
        elif args.operation == "report":
            from reclab_agent.domain.configuration import read_json_strict
            from reclab_agent.reporting import format_user_summary
            from reclab_agent.reporting import write_user_report
            saved = read_json_strict(args.input, max_bytes=32_000_000)
            if not isinstance(saved, dict):
                raise ValueError("saved experiment report must be a JSON object")
            output = args.output_directory if args.output_directory is not None else args.input.parent
            context = dict(saved["context"]) if isinstance(saved.get("context"), dict) else {}
            context["artifact_directory"] = args.input.absolute().parent
            context["report_json_path"] = args.input.absolute()
            generated = write_user_report(output, saved, context=context)
            if args.json:
                print(canonical_json({"ok": True, "result": generated}))
            else:
                print(format_user_summary({**saved, **generated}))
            return 0
        elif not internal and args.command in {"plan", "api-check", "run", "resume"}:
            from reclab_agent.application.launch import check_agent_api
            from reclab_agent.application.launch import prepare_launch
            from reclab_agent.application.launch import run_optimization
            if args.operation == "plan":
                config, plan, _ = prepare_launch(args.config)
                value = {"plan": plan.to_dict(), "plan_hash": plan.plan_hash,
                         "output_directory": str(config.output_directory), "api_requested": False,
                         "training_started": False}
            elif args.operation == "api-check":
                value = check_agent_api(args.config, env_file=args.env_file)
            else:
                value = run_optimization(args.config, env_file=args.env_file,
                    approve_plan=args.approve_plan, windows_ssh=args.windows_ssh,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                    resume_directory=args.session if args.operation == "resume" else None)
                result = {"ok": value["status"] == "completed", "result": value}
                if args.json:
                    print(canonical_json(result))
                else:
                    from reclab_agent.reporting import format_user_summary
                    print(format_user_summary(value))
                return 0 if result["ok"] else 2
        elif args.operation == "prepare":
            value = demo.prepare(args.root)
        elif args.operation == "status":
            value = demo.status(args.root)
        elif args.operation == "doctor":
            worker, _, _ = demo.worker_for(args.root, args.image)
            value = worker.capabilities()
        elif args.operation == "run":
            result = run_supervised(args.root, args.image)
            print(canonical_json(result))
            return 0 if result.get("ok") else 2
        elif args.operation == "_execute":
            value = demo.execute(args.root, args.image)
        else:
            value = demo.cancel_and_confirm(args.root, args.image)
        result = {"ok": True, "result": value}
    except KeyboardInterrupt:
        print("已中断；已启动实验的报告保留在本轮结果目录。", file=sys.stderr)
        return 130
    except (CapabilityError, DockerError, ValueError, KeyError, OSError, TimeoutError, sqlite3.Error) as error:
        result = _error(error)
    print(canonical_json(result))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
