"""A small user entrypoint that resolves setup details before the existing run."""
from __future__ import annotations
import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.execution.locking import job_lock
from reclab_agent.infrastructure.files import atomic_write
from reclab_agent.domain.configuration import load_task_configuration, read_json_strict
from reclab_agent.application.launch import LaunchError, prepare_launch, run_optimization
from reclab_agent.reporting import format_user_summary


def load_profile(root, config=None, env_file=None):
    """Read paths only. This function never reads the credential file itself."""
    root = Path(root).absolute()
    profile_path = root / "reclab.json"
    if profile_path.exists():
        profile = read_json_strict(profile_path, max_bytes=16_384)
        if not isinstance(profile, dict) or profile.keys() != {"task", "env_file"}:
            raise ValueError("reclab.json must contain task and env_file paths only")
    elif config is not None:
        profile = {"task": config, "env_file": ".reclab/agent.env"}
    else:
        raise ValueError("reclab.json is missing from the project directory")

    def resolve(value):
        if not isinstance(value, (str, Path)) or not str(value).strip() or any(c in str(value) for c in "\x00\r\n"):
            raise ValueError("startup configuration paths must be nonempty literal paths")
        path = Path(value)
        return Path(os.path.abspath(path if path.is_absolute() else root / path))

    return resolve(config if config is not None else profile["task"]), resolve(
        env_file if env_file is not None else profile["env_file"])


def _usable_python(path, *, web=False):
    try:
        check = ("import sys,fastapi,uvicorn,jinja2; assert sys.version_info >= (3,10)" if web else
                 "import sys,openai; assert sys.version_info >= (3,10); assert int(openai.__version__.split('.')[0]) >= 2")
        result = subprocess.run([str(path), "-I", "-c", check],
            capture_output=True, timeout=30, check=False)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ensure_runtime(root, argv):
    """Reuse an SDK environment, preparing one once when it is missing."""
    web = "--web" in argv
    try:
        if web and all(importlib.util.find_spec(name) is not None for name in ("fastapi", "uvicorn", "jinja2")):
            return None
        if not web and importlib.util.find_spec("openai") is not None and int(importlib.metadata.version("openai").split(".")[0]) >= 2:
            return None
    except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
        pass
    root = Path(root)
    state = root / ".reclab" / "controller.json"
    try:
        cached = read_json_strict(state, max_bytes=16_384)
        path = Path(cached["python"])
        if path.is_absolute() and path != Path(sys.executable) and _usable_python(path, web=web):
            return subprocess.call([str(path), "-B", "-m", "reclab_agent.application.start", *argv], cwd=root)
    except (OSError, ValueError, TypeError, KeyError):
        pass
    # WSL's /tmp can disappear between invocations. Keep its environment on the
    # persistent Linux filesystem rather than the slow mounted Windows drive.
    if os.name == "nt":
        environment = root / ".reclab" / "controller"
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        environment = cache / "reclab-agent" / f"python-{sys.version_info.major}.{sys.version_info.minor}"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not _usable_python(python, web=web):
        print("Preparing the local Agent environment (first start only)...", flush=True)
        create = subprocess.run([sys.executable, "-m", "venv", str(environment)],
                                capture_output=True, check=False)
        if create.returncode:
            raise LaunchError("could not prepare the local Python environment")
        dependencies = ["fastapi>=0.115,<1", "uvicorn>=0.30,<1", "jinja2>=3.1,<4"] if web else ["openai>=2.0"]
        install = subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                                  "--no-compile", *dependencies], capture_output=True, check=False)
        if install.returncode or not _usable_python(python, web=web):
            raise LaunchError("could not install the Agent SDK; check network access and retry")
    atomic_write(state, canonical_json({"python": str(python)}).encode("utf-8"))
    return subprocess.call([str(python), "-B", "-m", "reclab_agent.application.start", *argv], cwd=root)


def _progress(message):
    translations = {
        "Checking Responses API and tool-call compatibility": "正在检查模型服务...",
        "Training and independently scoring the unmodified baseline": "正在训练并评分基线...",
    }
    if message.startswith("Starting the optimization Agent;"):
        message = "Agent 正在分析模型并验证候选..."
    elif message.startswith("Session: "):
        message = "本轮结果目录：" + message.removeprefix("Session: ")
    elif message.startswith("Baseline metrics: "):
        message = "基线指标：" + message.removeprefix("Baseline metrics: ")
    elif message.startswith("User report: "):
        message = "结果报告：" + message.removeprefix("User report: ")
    print(translations.get(message, message), flush=True)


def _scope(config):
    budget = config.task.budget
    print("任务：" + config.task.question, flush=True)
    print(f"范围：最多 {budget.max_training_jobs} 次训练，单次实验 {budget.max_job_seconds} 秒，"
          f"{budget.max_patches} 次修改。", flush=True)


def main(argv=None, *, project_root=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Start the configured model-optimization Agent")
    parser.add_argument("--plan", action="store_true", help="view the task without API calls or training")
    parser.add_argument("--web", action="store_true", help="open the local workbench without loading API keys or starting training")
    parser.add_argument("--port", type=int, default=8000, help="local workbench port")
    parser.add_argument("--config", help="override the task selected in reclab.json")
    parser.add_argument("--env-file", help="override the local API settings file")
    parser.add_argument("--windows-ssh", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    try:
        if args.web:
            if args.plan or args.config or args.env_file:
                raise ValueError("--web cannot be combined with --plan, --config or --env-file")
            delegated = ensure_runtime(root, argv)
            if delegated is not None:
                return delegated
            from reclab_agent.cli.web import main as web_main
            web_main(["--root", str(root), "--port", str(args.port)])
            return 0
        task_path, env_path = load_profile(root, args.config, args.env_file)
        config = load_task_configuration(task_path)
        if args.plan:
            _, plan, _ = prepare_launch(config)
            _scope(config)
            print("任务配置：" + str(task_path))
            print("允许修改：" + ", ".join(plan.editable_files))
            print("结果目录：" + str(config.output_directory))
            return 0
        delegated = ensure_runtime(root, argv)
        if delegated is not None:
            return delegated
        _scope(config)
        # Starting this command is the user's explicit instruction to execute
        # the displayed, configured scope. The model never grants approval.
        with job_lock(config.output_directory, "optimization-start") as acquired:
            if not acquired:
                raise LaunchError("this task is already running; wait for its current session")
            report = run_optimization(config, env_file=env_path, approve_plan=True,
                                      windows_ssh=args.windows_ssh, progress=_progress)
        status = report["status"]
        print(format_user_summary(report), flush=True)
        return 0 if status == "completed" else 2
    except KeyboardInterrupt:
        print("已中断，实验状态保留在本地结果目录。", file=sys.stderr)
        return 130
    except (ValueError, OSError) as error:
        print("启动失败：" + str(error), file=sys.stderr)
        return 2
    except Exception as error:
        print("启动失败：" + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
