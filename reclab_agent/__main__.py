from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .benchmark import load_tasks, run_reference_suite
from .tools import RecLabToolbox


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RecLab deterministic tool benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="RecLab Agent repository root")
    parser.add_argument("--tasks", type=Path, help="Task manifest; defaults to tasks/dev.json")
    parser.add_argument("--list-tools", action="store_true", help="Print tool schemas instead of running tasks")
    args = parser.parse_args()

    root = Path(os.path.abspath(os.fspath(args.root)))
    toolbox = RecLabToolbox(root)
    if args.list_tools:
        payload = [{"name": spec.name, "description": spec.description,
                    "input_schema": spec.input_schema} for spec in toolbox.specs]
    else:
        task_path = args.tasks or root / "tasks" / "dev.json"
        payload = run_reference_suite(toolbox, load_tasks(task_path))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

