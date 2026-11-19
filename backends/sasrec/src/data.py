"""Read the public RecLab next-item bundle and target-model configuration.

This module is self-contained and uses only the standard library. ``--data``
is the directory containing manifest.json, train.json and valid.json. No raw
prepared dataset or private test file is opened. Runtime isolation belongs to
the Worker; this loader is a data-contract check, not an execution sandbox.
"""
from __future__ import annotations

import hashlib
import json
import keyword
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any


CONFIG_KEYS = frozenset({
    "model", "max_length", "epochs", "patience", "batch_size",
    "eval_batch_size", "learning_rate", "weight_decay", "grad_clip", "device",
})


def canonical_json(value: Any) -> str:
    """Match reclab_agent.contracts.canonical_json without importing the host."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


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


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_object,
                      parse_float=_finite_float,
                      parse_constant=_reject_constant)


def load_public_data(path: str | Path) -> dict[str, Any]:
    """Validate and preserve the public bundle; return its canonical identity.

    Training repetitions and ordering are retained. Validation uses exactly the
    supplied users; malformed rows fail instead of silently changing the cohort.
    The resulting data_hash is identical to DatasetBundle.data_hash.
    """
    public = Path(path)
    if public.is_symlink() or not public.is_dir():
        raise ValueError("data must be the public bundle directory")
    manifest = _read_json(public / "manifest.json")
    train = _read_json(public / "train.json")
    valid = _read_json(public / "valid.json")
    if (not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("task") != "next_item"):
        raise ValueError("expected RecLab schema 1 next_item public data")
    dataset_id = manifest.get("dataset_id")
    if (not isinstance(dataset_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", dataset_id) is None):
        raise ValueError("invalid public dataset_id")
    catalog = manifest.get("catalog")
    if (not isinstance(catalog, list) or not catalog
            or any(type(item) is not int for item in catalog)
            or catalog != list(range(1, len(catalog) + 1))):
        raise ValueError("next-item backend requires an ordered contiguous catalog from 1 to N")
    num_items = len(catalog)
    if not isinstance(train, dict) or not train:
        raise ValueError("train must be a nonempty mapping of users to histories")
    for user, history in train.items():
        if (not isinstance(user, str) or not user.strip()
                or not isinstance(history, list)
                or any(type(item) is not int or not 1 <= item <= num_items
                       for item in history)):
            raise ValueError("invalid public training history")
    if not isinstance(valid, dict) or not valid:
        raise ValueError("valid must be a nonempty mapping of users to validation rows")
    for user, row in valid.items():
        if (user not in train or not isinstance(row, dict)
                or set(row) != {"history", "target"}):
            raise ValueError("invalid public validation row")
        history, target = row["history"], row["target"]
        if (not isinstance(history, list)
                or any(type(item) is not int or not 1 <= item <= num_items
                       for item in history)
                or history != train[user]
                or type(target) is not int or not 1 <= target <= num_items
                or target in history):
            raise ValueError("validation history/target violates next-item protocol")
    counts = manifest.get("counts", {})
    if not isinstance(counts, dict):
        raise ValueError("public manifest counts must be an object")
    actual_counts = {"train_users": len(train), "valid_users": len(valid),
                     "train_interactions": sum(len(history) for history in train.values())}
    for key, actual in actual_counts.items():
        if key in counts and (type(counts[key]) is not int or counts[key] != actual):
            raise ValueError(f"public manifest {key} disagrees with data")
    payload = {"manifest": manifest, "train": train, "valid": valid}
    return {**payload, "num_items": num_items,
            "data_hash": hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load model selection and training settings; never select CPU implicitly.

    The runtime injects num_items and max_length into the selected model
    constructor. Constructor-specific kwargs remain the model's responsibility.
    """
    config = _read_json(Path(path))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("config must contain exactly the documented configuration keys")
    model = config["model"]
    if not isinstance(model, dict) or set(model) != {"file", "class", "kwargs"}:
        raise ValueError("model must contain exactly file, class and kwargs")
    model_file = model["file"]
    if (not isinstance(model_file, str) or "\\" in model_file or "\x00" in model_file
            or not model_file.startswith("src/")
            or PurePosixPath(model_file).is_absolute() or PureWindowsPath(model_file).drive
            or any(part in {"", ".", ".."} for part in model_file.split("/"))
            or PurePosixPath(model_file).suffix != ".py"):
        raise ValueError("model.file must be a relative src/ Python file without traversal")
    class_name = model["class"]
    if (not isinstance(class_name, str) or not class_name.isidentifier()
            or keyword.iskeyword(class_name)):
        raise ValueError("model.class must be a Python class identifier")
    kwargs = model["kwargs"]
    if (not isinstance(kwargs, dict)
            or any(not key.isidentifier() or keyword.iskeyword(key) for key in kwargs)
            or {"num_items", "max_length"} & kwargs.keys()):
        raise ValueError("model.kwargs keys must be identifiers and cannot override num_items or max_length")
    minimums = {"max_length": 1, "epochs": 1, "patience": 0,
                "batch_size": 1, "eval_batch_size": 1}
    for key, minimum in minimums.items():
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("learning_rate", "weight_decay", "grad_clip"):
        value = config[key]
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{key} must be a finite number")
    if config["learning_rate"] <= 0 or config["grad_clip"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("learning_rate and grad_clip must be positive; weight_decay must be nonnegative")
    if config["device"] not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or explicit cpu")
    return config
