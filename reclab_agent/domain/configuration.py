"""Strict, path-based research task configuration; loading never runs a model."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any
from reclab_agent.domain.contracts import Budget, TaskSpec, identifier
from reclab_agent.infrastructure.files import read_regular_file


@dataclass(frozen=True)
class TaskConfiguration:
    configuration_path: Path
    task: TaskSpec
    source_directory: Path
    config_name: str
    editable_files: tuple[str, ...]
    dataset_directory: Path
    dataset_adapter: str
    expected_data_hash: str | None
    evaluator_id: str
    primary_metric: str
    guardrail_metrics: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    model_env: str
    api_key_env: str
    execution_backend: str
    execution_policy_path: Path | None
    output_directory: Path
    execution_connection_path: Path | None = None
    baseline_report: Path | None = None
    project_directory: Path | None = None
    max_wall_seconds: int | None = None
    max_network_requests: int = 10
    tavily_key_env: str = "TAVILY_API_KEY"
    resume_jobs: bool = False
    retrieval: dict | None = None

    @property
    def model_config_path(self) -> Path:
        return self.source_directory / "configs" / f"{self.config_name}.json"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON number")
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_json_strict(path: str | Path, max_bytes: int = 512_000_000) -> Any:
    """Read bounded ordinary JSON, rejecting duplicate keys and nonfinite values."""
    raw = read_regular_file(Path(path), max_bytes=max_bytes)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_float=_finite_float, parse_constant=_reject_constant)


def _object(value: Any, required: set[str], context: str,
            optional: set[str] | None = None) -> dict[str, Any]:
    if (not isinstance(value, dict) or not required <= value.keys()
            or not value.keys() <= required | (optional or set())):
        raise ValueError(f"{context} must contain the documented fields only")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be nonempty text")
    return value


def _array(value: Any, context: str, *, nonempty: bool = True) -> tuple:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{context} must be {'a nonempty' if nonempty else 'an'} array")
    # Equality-based duplicate checks also reject arrays containing unhashable
    # malformed values with a normal configuration error instead of TypeError.
    if any(item in value[:index] for index, item in enumerate(value)):
        raise ValueError(f"{context} must contain unique values")
    return tuple(value)


def _path(value: Any, parent: Path, context: str) -> Path:
    value = _text(value, context)
    if any(char in value for char in ("\x00", "\n", "\r")) or value.startswith("~"):
        raise ValueError(f"{context} must be a literal filesystem path")
    windows = PureWindowsPath(value)
    if os.name != "nt" and (windows.drive or "\\" in value):
        raise ValueError(f"{context} uses a Windows path on a non-Windows host")
    path = Path(value)
    # abspath deliberately preserves existing symlink components: inspection
    # must see and reject them rather than silently resolving their targets.
    return Path(os.path.abspath(path if path.is_absolute() else parent / path))


def _source_path(value: Any) -> str:
    value = _text(value, "source.editable_files entry")
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    if ("\\" in value or "\x00" in value or posix.is_absolute() or windows.drive
            or not value.startswith(("src/", "configs/"))
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or posix.suffix not in {".py", ".json", ".yaml", ".yml"}):
        raise ValueError("editable files must be relative src/configs text paths")
    return value


def _environment_name(value: Any, context: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) is None:
        raise ValueError(f"{context} must name an environment variable")
    return value


def load_task_configuration(path: str | Path) -> TaskConfiguration:
    """Parse user paths and task scope without creating directories or loading code.

    Relative asset paths are based on the configuration file's parent directory,
    independent of the caller's working directory. Registry IDs are plain names;
    the configuration cannot supply commands, import paths or credential values.
    """
    configuration_path = _path(os.fspath(path), Path.cwd(), "configuration path")
    value = _object(read_json_strict(configuration_path, max_bytes=1_000_000), {
        "schema_version", "task", "source", "data", "evaluation", "experiment",
        "agent", "execution", "output",
    }, "task configuration", {"research"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported task configuration schema_version")
    parent = configuration_path.parent
    task_value = _object(value["task"], {"task_id", "question", "dataset_id", "baseline_id"},
                         "task", {"budget", "mode", "schema_version"})
    if "schema_version" in task_value and (type(task_value["schema_version"]) is not int
                                            or task_value["schema_version"] != 1):
        raise ValueError("unsupported task schema_version")
    budget_value = _object(task_value.get("budget", {}), set(), "task.budget", {
        "max_model_turns", "max_patches", "max_training_jobs", "max_job_seconds",
        "max_total_job_seconds", "max_cost_cny",
    })
    try:
        budget = Budget(**budget_value)
        task = TaskSpec(**{**task_value, "budget": budget})
    except (TypeError, OverflowError) as error:
        raise ValueError("invalid task or task budget") from error

    source = _object(value["source"], {"directory", "config", "editable_files"}, "source")
    config_name = identifier(source["config"])
    editable_files = tuple(_source_path(item)
                           for item in _array(source["editable_files"], "source.editable_files"))
    data = _object(value["data"], {"directory", "adapter", "expected_hash"}, "data")
    dataset_adapter = identifier(data["adapter"])
    expected_hash = data["expected_hash"]
    if expected_hash is not None and (not isinstance(expected_hash, str)
                                      or re.fullmatch(r"[a-f0-9]{64}", expected_hash) is None):
        raise ValueError("data.expected_hash must be a lowercase SHA256 digest or null")
    evaluation = _object(value["evaluation"], {"evaluator", "primary_metric", "guardrail_metrics"},
                         "evaluation")
    evaluator_id = identifier(evaluation["evaluator"])
    primary_metric = _text(evaluation["primary_metric"], "evaluation.primary_metric")
    guardrails = tuple(_text(item, "evaluation.guardrail_metrics entry")
                       for item in _array(evaluation["guardrail_metrics"],
                                          "evaluation.guardrail_metrics", nonempty=False))
    experiment = _object(value["experiment"], {"candidate_ids", "seeds"}, "experiment", {"baseline_report"})
    candidate_ids = tuple(identifier(item) for item in
                          _array(experiment["candidate_ids"], "experiment.candidate_ids"))
    seeds = _array(experiment["seeds"], "experiment.seeds")
    if any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("experiment.seeds must contain nonnegative 31-bit integers")
    agent = _object(value["agent"], {"model_env", "api_key_env"}, "agent")
    model_env = _environment_name(agent["model_env"], "agent.model_env")
    api_key_env = _environment_name(agent["api_key_env"], "agent.api_key_env")
    execution = _object(value["execution"], {"backend"}, "execution", {"policy", "connection"})
    policy = connection = None
    if execution["backend"] == "docker":
        _object(execution, {"backend", "policy"}, "execution")
        policy = (None if execution["policy"] is None else
                  _path(execution["policy"], parent, "execution.policy"))
    elif execution["backend"] == "ssh_tmux":
        _object(execution, {"backend", "connection"}, "execution")
        connection = _path(execution["connection"], parent, "execution.connection")
    else:
        raise ValueError("execution.backend must be docker or ssh_tmux")
    research = _object(value.get("research", {}), set(), "research",
                       {"project_directory", "max_wall_seconds", "max_network_requests", "tavily_key_env", "retrieval"})
    retrieval = research.get("retrieval")
    if retrieval is not None:
        _object(retrieval, {"mode"}, "research.retrieval", {"embedding_model", "embedding_revision",
                "reranking_model", "reranking_revision", "allow_download"})
        if retrieval["mode"] not in {"bm25", "dense", "hybrid", "reranked"}:
            raise ValueError("invalid research.retrieval.mode")
        for key in ("embedding_model", "embedding_revision", "reranking_model", "reranking_revision"):
            if key in retrieval:
                _text(retrieval[key], "research.retrieval." + key)
        if "allow_download" in retrieval and type(retrieval["allow_download"]) is not bool:
            raise ValueError("research.retrieval.allow_download must be boolean")
    for name in ("max_wall_seconds", "max_network_requests"):
        if name in research and (type(research[name]) is not int or research[name] < 1):
            raise ValueError("research limits must be positive integers")
    if "tavily_key_env" in research:
        _environment_name(research["tavily_key_env"], "research.tavily_key_env")
    output = _object(value["output"], {"directory"}, "output")
    return TaskConfiguration(
        configuration_path=configuration_path, task=task,
        source_directory=_path(source["directory"], parent, "source.directory"),
        config_name=config_name, editable_files=editable_files,
        dataset_directory=_path(data["directory"], parent, "data.directory"),
        dataset_adapter=dataset_adapter, expected_data_hash=expected_hash,
        evaluator_id=evaluator_id, primary_metric=primary_metric, guardrail_metrics=guardrails,
        candidate_ids=candidate_ids, seeds=seeds, model_env=model_env, api_key_env=api_key_env,
        execution_backend=execution["backend"], execution_policy_path=policy,
        output_directory=_path(output["directory"], parent, "output.directory"),
        execution_connection_path=connection,
        project_directory=_path(research["project_directory"], parent, "research.project_directory") if research.get("project_directory") else None,
        max_wall_seconds=research.get("max_wall_seconds"),
        max_network_requests=research.get("max_network_requests", 10),
        tavily_key_env=research.get("tavily_key_env", "TAVILY_API_KEY"),
        retrieval=retrieval,
        baseline_report=_path(experiment["baseline_report"], parent, "experiment.baseline_report") if experiment.get("baseline_report") else None,
    )
