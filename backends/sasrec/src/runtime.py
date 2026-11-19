"""Shared candidate-runtime operations; imported only inside experiment jobs."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import sys

import torch

from data import canonical_json


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def implementation_hash():
    source = Path(__file__).resolve().parent
    return digest({path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(source.rglob("*.py"))})


def model_config(config, num_items):
    return {"file": config["model"]["file"], "class": config["model"]["class"],
            "kwargs": {"num_items": num_items, "max_length": config["max_length"],
                       **config["model"]["kwargs"]}}


def build_model(config, num_items):
    """Load the registered model inside the job; never used by the controller."""
    description = model_config(config, num_items)
    source = Path(__file__).resolve().parent
    path = (source.parent / description["file"]).resolve()
    if source not in path.parents or not path.is_file():
        raise ValueError("configured model file must exist inside the snapshot src directory")
    spec = importlib.util.spec_from_file_location("_reclab_target_model", path)
    if spec is None or spec.loader is None:
        raise ValueError("configured model file cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    constructor = getattr(module, description["class"], None)
    if not callable(constructor):
        raise ValueError("configured model class was not found")
    model = constructor(**description["kwargs"])
    if (not isinstance(model, torch.nn.Module)
            or type(getattr(model, "max_length", None)) is not int
            or model.max_length != config["max_length"]
            or not callable(getattr(model, "encode", None))
            or not callable(getattr(model, "catalog_logits", None))):
        raise ValueError("model must implement the next-item torch Module interface: max_length, encode, catalog_logits")
    return model


def initialize(config):
    try:
        seed = int(os.environ["RECLAB_SEED"])
    except (KeyError, ValueError) as error:
        raise ValueError("RECLAB_SEED must be supplied by the experiment runner") from error
    if not 0 <= seed < 2**31:
        raise ValueError("RECLAB_SEED must be a nonnegative 31-bit integer")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    device = torch.device(config["device"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return device, seed


def histories_tensor(histories, max_length, device):
    if any(not row for row in histories):
        raise ValueError("next-item model inputs must contain at least one observed item")
    rows = [[0] * max(0, max_length - len(row)) + row[-max_length:] for row in histories]
    return torch.tensor(rows, dtype=torch.long, device=device)


@torch.inference_mode()
def predict_valid(model, data, device, batch_size):
    """Rank the whole catalog; mask the complete history, not its truncated input."""
    model.eval()
    users = sorted(data["valid"])
    predictions = {}
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        histories = [data["valid"][user]["history"] for user in batch_users]
        scores = model.catalog_logits(model.encode(histories_tensor(histories, model.max_length, device)))
        if scores.shape != (len(batch_users), data["num_items"]):
            raise ValueError("model scores must have shape [batch, catalog_items]")
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("nonfinite prediction logits; predictions were not accepted")
        for index, history in enumerate(histories):
            seen = sorted(set(history))
            if seen:
                scores[index, torch.tensor(seen, device=device) - 1] = -torch.inf
        # Per-user k also supports tiny catalogs without emitting masked items.
        top = (scores.topk(min(20, data["num_items"]), dim=-1).indices + 1).cpu().tolist()
        for user, history, ranked in zip(batch_users, histories, top):
            predictions[user] = ranked[:min(20, data["num_items"] - len(set(history)))]
    return predictions


def validation_ndcg20(predictions, valid):
    """Training-only checkpoint selection; official scores come from the host grader."""
    values = []
    for user, row in valid.items():
        ranked = predictions[user]
        values.append(1.0 / math.log2(ranked.index(row["target"]) + 2)
                      if row["target"] in ranked else 0.0)
    return math.fsum(values) / len(values)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path, data, config, seed):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"format": "reclab-next-item-public-v1", "data_hash": data["data_hash"],
                "config_hash": digest(config), "seed": seed,
                "implementation_hash": implementation_hash(),
                "model_config": model_config(config, data["num_items"])}
    if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint does not match the source, public data, configuration and seed")
    if type(saved.get("epoch")) is not int or not 1 <= saved["epoch"] <= config["epochs"]:
        raise ValueError("checkpoint has an invalid selected epoch")
    score = saved.get("selection_valid_ndcg@20")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("checkpoint has an invalid selection score")
    model = build_model(config, data["num_items"])
    model.load_state_dict(saved["model"], strict=True)
    return model, saved
