"""Administrator CLI for the Docker worker; run on the Linux daemon host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import canonical_json
from .execution import CapabilityError, DockerError, DockerWorker, InputTree, JobSpec, ResourceLimits, WorkerPolicy
from ._execution.config import load_policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, help="private durable worker directory")
    parser.add_argument("--policy", required=True, help="administrator-approved JSON policy")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("recover")
    serve = commands.add_parser("serve")
    serve.add_argument("--interval", type=float, default=1.0)
    submit = commands.add_parser("submit")
    submit.add_argument("--spec", required=True, help="JSON JobSpec, not an executable command")
    for name in ("status", "cancel", "collect", "logs"):
        command = commands.add_parser(name)
        command.add_argument("job_id")
    args = parser.parse_args(argv)
    try:
        worker = DockerWorker(args.state, load_policy(args.policy))
        if args.command == "doctor":
            result = worker.capabilities()
        elif args.command == "recover":
            result = worker.recover()
        elif args.command == "serve":
            worker.serve(interval=args.interval)
            result = {"supervisor": "stopped"}
        elif args.command == "submit":
            result = worker.submit(JobSpec.from_dict(json.loads(Path(args.spec).read_text(encoding="utf-8"))))
        elif args.command == "status":
            result = worker.poll(args.job_id)
        elif args.command == "cancel":
            result = worker.cancel(args.job_id)
        elif args.command == "collect":
            result = worker.collect_predictions(args.job_id)
        else:
            result = {"logs": worker.logs(args.job_id)}
        print(canonical_json({"ok": True, "result": result}))
        return 0
    except (CapabilityError, DockerError, ValueError, KeyError, OSError) as error:
        print(canonical_json({"ok": False, "error": type(error).__name__, "detail": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
