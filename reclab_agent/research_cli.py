"""Week-two controlled fixture demo. Candidate code runs exclusively in Docker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from . import demo
from .contracts import canonical_json
from .execution import CapabilityError, DockerError


def _error(error: BaseException) -> dict:
    return {"ok": False, "error": type(error).__name__, "detail": str(error), "host_fallback": False}


def _child(root: str, image: str, operation: str, timeout: int) -> dict:
    # This invokes trusted controller code, never src/train.py or src/predict.py.
    completed = subprocess.run([sys.executable, "-m", "reclab_agent.research_cli", operation,
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
        commands = commands.add_parser("demo", help="audited fixture, not an LLM research agent").add_subparsers(dest="operation", required=True)
        for name in ("prepare", "status", "doctor", "run"):
            command = commands.add_parser(name)
            command.add_argument("--root", required=True, help="new directory for prepare; prepared root otherwise")
            if name in {"doctor", "run"}:
                command.add_argument("--image", required=True, help="pre-pulled administrator-approved repository@sha256:digest")
    args = parser.parse_args(argv)
    try:
        if args.operation == "prepare":
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
    except (CapabilityError, DockerError, ValueError, KeyError, OSError, TimeoutError, sqlite3.Error) as error:
        result = _error(error)
    print(canonical_json(result))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
