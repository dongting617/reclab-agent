"""Inspect a configured task without importing candidate code or starting jobs.

Adapter callbacks are trusted controller code registered by the host. A JSON
configuration selects their names; it cannot import a Python model or scorer.
Successful input inspection is deliberately separate from execution readiness.
"""
from __future__ import annotations
import ast
import importlib.util
import keyword
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
from typing import Any
from reclab_agent.domain.contracts import canonical_json, validate_digest
from reclab_agent.execution.worker import InputTree
from reclab_agent.infrastructure.files import confined_path, read_regular_file
from reclab_agent.domain.configuration import TaskConfiguration, read_json_strict
from reclab_agent.experiments.adapters import TaskAdapterRegistry


def _source_model_config(config: TaskConfiguration) -> dict[str, Any]:
    value = read_json_strict(config.model_config_path, max_bytes=1_000_000)
    if not isinstance(value, dict):
        raise ValueError("model configuration must be a JSON object")
    model = value.get("model")
    if not isinstance(model, dict) or set(model) != {"file", "class", "kwargs"}:
        raise ValueError("model configuration must define model.file, model.class and model.kwargs")
    model_file = model["file"]
    if (not isinstance(model_file, str) or not model_file.startswith("src/")
            or "\\" in model_file or "\x00" in model_file
            or PurePosixPath(model_file).is_absolute() or PureWindowsPath(model_file).drive
            or any(part in {"", ".", ".."} for part in model_file.split("/"))
            or PurePosixPath(model_file).suffix != ".py"):
        raise ValueError("model.file must be a relative Python file inside src")
    class_name = model["class"]
    if (not isinstance(class_name, str) or not class_name.isidentifier()
            or keyword.iskeyword(class_name)):
        raise ValueError("model.class must be a Python class identifier")
    kwargs = model["kwargs"]
    if (not isinstance(kwargs, dict)
            or any(not key.isidentifier() or keyword.iskeyword(key) for key in kwargs)
            or {"num_items", "max_length"} & kwargs.keys()):
        raise ValueError("model.kwargs cannot override num_items/max_length and keys must be identifiers")
    path = confined_path(config.source_directory, model_file)
    tree = ast.parse(read_regular_file(path, max_bytes=1_000_000).decode("utf-8"), filename=model_file)
    if class_name not in {node.name for node in tree.body if isinstance(node, ast.ClassDef)}:
        raise ValueError("configured model class must be declared in the configured model file")
    for key, minimum in (("max_length", 1), ("epochs", 1), ("patience", 0),
                         ("batch_size", 1), ("eval_batch_size", 1)):
        if key in value and (type(value[key]) is not int or value[key] < minimum):
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key, allow_zero in (("learning_rate", False), ("weight_decay", True), ("grad_clip", False)):
        if key not in value:
            continue
        number = value[key]
        try:
            finite = type(number) in (int, float) and math.isfinite(number)
        except OverflowError:
            finite = False
        if not finite or number < 0 or (not allow_zero and number == 0):
            raise ValueError(f"invalid numeric training setting: {key}")
    if "device" in value and value["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be explicit cpu or cuda")
    return value


def _source(config: TaskConfiguration) -> dict[str, Any]:
    frozen = InputTree.freeze(config.source_directory, "source")
    files = sorted(path.relative_to(config.source_directory).as_posix()
                   for path in config.source_directory.rglob("*") if path.is_file())
    if not set(config.editable_files) <= set(files):
        raise ValueError("editable_files contains files absent from the source tree")
    for name in files:
        path = confined_path(config.source_directory, name)
        if path.suffix == ".py":
            ast.parse(read_regular_file(path, max_bytes=1_000_000).decode("utf-8"), filename=name)
        elif path.suffix == ".json":
            read_json_strict(path, max_bytes=1_000_000)
    model_config = _source_model_config(config)
    frozen.verify()
    return {"directory": str(config.source_directory), "tree_sha256": frozen.sha256,
            "files": files, "editable_files": list(config.editable_files),
            "readonly_files": sorted(set(files) - set(config.editable_files)),
            "config": config.config_name, "model": model_config["model"],
            "source_config_validation": "static_model_spec_and_present_common_fields",
            "model_constructor_verified": False, "candidate_runtime_verified": False}


def _reject_link_components(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("configured paths must not contain symlink/reparse components")


def _paths(config: TaskConfiguration) -> dict[str, Any]:
    _reject_link_components(config.output_directory)
    if config.execution_policy_path is not None:
        _reject_link_components(config.execution_policy_path)
    if config.execution_connection_path is not None:
        _reject_link_components(config.execution_connection_path)
    output = config.output_directory.resolve()
    if config.project_directory is not None:
        project = config.project_directory.resolve()
        _reject_link_components(config.project_directory)
        for protected in (config.source_directory.resolve(), config.dataset_directory.resolve()):
            if project == protected or project in protected.parents or protected in project.parents:
                raise ValueError("project_directory must not overlap source or dataset")
    for label, value in (("source", config.source_directory), ("dataset", config.dataset_directory)):
        protected = value.resolve()
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError(f"output_directory must not overlap the {label} directory")
    if output.exists() and not output.is_dir():
        raise ValueError("output_directory exists but is not a directory")
    return {"output_directory": str(config.output_directory), "output_created": False}


def inspect_task(config: TaskConfiguration, *, registry: TaskAdapterRegistry | None = None) -> dict[str, Any]:
    """Return input checks and unmet runtime requirements; perform no execution."""
    registry = registry if registry is not None else TaskAdapterRegistry()
    report: dict[str, Any] = {
        "configuration_path": str(config.configuration_path), "task": config.task.to_dict(),
        "inputs_valid": False, "ready_to_run": False, "training_started": False,
        "model_imported": False, "test_labels_read": False, "checks": [], "errors": [],
        "experiment_tool": {"implementation": "train_predict_score_v1", "available": True, "bound": False},
        "pending": ["experiment_runner_not_connected", "isolated_numerical_acceptance_required",
                    "research_plan_review_required"],
    }

    def check(name, callback):
        try:
            result = callback()
        except (OSError, ValueError, TypeError, SyntaxError, KeyError, AttributeError) as error:
            report["checks"].append({"name": name, "ok": False})
            report["errors"].append({"check": name, "error": type(error).__name__, "detail": str(error)})
            return None
        report["checks"].append({"name": name, "ok": True})
        return result

    paths = check("paths", lambda: _paths(config))
    if paths is not None:
        report["paths"] = paths
    source = check("source", lambda: _source(config))
    if source is not None:
        report["source"] = source

    def load_data():
        # Freeze first: the loader's parsed identity and the eventual mounted
        # public tree must refer to the same bytes, including for custom loaders.
        expected_public = confined_path(config.dataset_directory, "public")
        public = InputTree.freeze(expected_public, "public_data")
        dataset = registry.load_dataset(config.dataset_adapter, config.dataset_directory)
        if Path(dataset.public).absolute() != expected_public.absolute():
            raise ValueError("dataset adapter must use the configured dataset/public directory")
        validate_digest(dataset.data_hash, "dataset data_hash")
        if dataset.dataset_id != config.task.dataset_id:
            raise ValueError("dataset identity does not match task.dataset_id")
        if config.expected_data_hash is not None and dataset.data_hash != config.expected_data_hash:
            raise ValueError("dataset data_hash differs from the configured expectation")
        info = dataset.info()
        if not isinstance(info, dict):
            raise TypeError("dataset info() must return a JSON object")
        canonical_json(info)
        public.verify()
        return dataset, public, {**info, "adapter": config.dataset_adapter,
                                 "directory": str(config.dataset_directory),
                                 "public_directory": str(expected_public),
                                 "public_tree_sha256": public.sha256,
                                 "data_hash": dataset.data_hash, "dataset_id": dataset.dataset_id,
                                 "identity_pinned": config.expected_data_hash is not None}

    loaded = check("dataset", load_data)
    if loaded is not None:
        dataset, public, info = loaded
        report["dataset"] = info

        def describe():
            description = registry.describe_evaluator(config.evaluator_id, dataset)
            public.verify()
            requested = {config.primary_metric, *config.guardrail_metrics}
            if not requested <= set(description.metrics):
                raise ValueError("requested metrics are not supported by the registered evaluator")
            return {"id": config.evaluator_id, "protocol_hash": description.protocol_hash,
                    "metrics": list(description.metrics), "primary_metric": config.primary_metric,
                    "guardrail_metrics": list(config.guardrail_metrics), "details": description.details,
                    "scoring_configured": registry.can_score(config.evaluator_id)}

        evaluation = check("evaluation", describe)
        if evaluation is not None:
            report["evaluation"] = evaluation
            if not evaluation["scoring_configured"]:
                report["pending"].append("evaluator_scoring_not_configured")
    if config.expected_data_hash is None:
        report["pending"].append("public_data_identity_not_pinned")

    try:
        sdk_present = importlib.util.find_spec("openai") is not None
    except (ImportError, ValueError):
        sdk_present = False
    model_present = bool(os.environ.get(config.model_env))
    credentials_present = bool(os.environ.get(config.api_key_env))
    report["agent_backend"] = {"sdk_installed": sdk_present, "model_env": config.model_env,
                               "model_configured": model_present, "api_key_env": config.api_key_env,
                               "credentials_configured": credentials_present,
                               "api_requested": False}
    for present, code in ((sdk_present, "agent_sdk_not_installed"), (model_present, "agent_model_not_configured"),
                          (credentials_present, "agent_credentials_not_configured")):
        if not present:
            report["pending"].append(code)
    policy = (config.execution_connection_path if config.execution_backend == "ssh_tmux"
              else config.execution_policy_path)
    policy_present = policy is not None and policy.is_file()
    report["execution"] = {"backend": config.execution_backend,
                           ("connection_path" if config.execution_backend == "ssh_tmux" else "policy_path"):
                               str(policy) if policy is not None else None,
                           ("connection_present" if config.execution_backend == "ssh_tmux" else "policy_present"):
                               policy_present, "capabilities_probed": False,
                           "worker_started": False}
    if not policy_present:
        report["pending"].append("ssh_connection_not_configured" if config.execution_backend == "ssh_tmux"
                                 else "execution_policy_not_configured")
    report["inputs_valid"] = not report["errors"]
    return report
