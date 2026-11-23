"""Trusted reproduction of ONE frozen B1 checkpoint on full Beauty validation.

This is an audit utility, never a candidate executor or a model-training tool.
All three source pins and the public bundle identity are hardcoded. It does not
read private test files, select a checkpoint, run shell commands, or train.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import math
from pathlib import Path
import platform
import sys
import time
import types
from typing import Any
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.data.datasets import DatasetBundle
from reclab_agent.memory.storage import atomic_write, confined_path


FROZEN_DATA_SHA256 = "7a2dae80bf7053c91d4eda79f33ff997df45c07c1fd968f36d77b591ed828efb"
FROZEN_CHECKPOINT_SHA256 = "828420f4664d4a78717a534165a2f2cd6902ff0543127f4b187b907b3a50813b"
FROZEN_MODEL_SOURCE_SHA256 = "37dfa3296117a22e19f6dadd2b2b31be5ee9f30c43febf108cd8911dc15ec1cb"
FROZEN_PUBLIC_DATA_HASH = "4a5b55b4de8ffe955b0c4dd4f1354abdba9e272e9595e1a21b5d2989046ee917"
FROZEN_MODEL_CONFIG = {"num_items": 12068, "max_length": 50, "hidden_size": 64,
                       "num_layers": 1, "num_heads": 1, "dropout": 0.4}
FROZEN_REFERENCE_NDCG20 = 0.07965987624038143
FROZEN_VALID_USERS = 22312


def pinned_bytes(path: str | Path, expected: str, label: str) -> bytes:
    """Return the exact verified bytes, avoiding a second unchecked file read."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{label} SHA256 differs from the hardcoded frozen pin")
    return raw


def require_new_output(output: str | Path) -> Path:
    output = Path(output)
    confined_path(output, "metadata.json")
    if output.exists():
        raise ValueError("output already exists; frozen reproduction never overwrites")
    return output


def verify_public_bundle(bundle: DatasetBundle) -> None:
    if bundle.data_hash != FROZEN_PUBLIC_DATA_HASH:
        raise ValueError("public bundle identity differs from the frozen Beauty D1 import")
    if (len(bundle.valid) != FROZEN_VALID_USERS or len(bundle.train) != 22363
            or bundle.manifest["catalog"] != list(range(1, 12069))):
        raise ValueError("public bundle population/catalog differs from the frozen contract")


def verify_checkpoint_metadata(saved: Any) -> None:
    if not isinstance(saved, dict):
        raise ValueError("frozen checkpoint must be a mapping")
    if saved.get("dataset_sha256") != FROZEN_DATA_SHA256:
        raise ValueError("checkpoint prepared-dataset identity differs")
    config = saved.get("model_config")
    if (not isinstance(config, dict) or config != FROZEN_MODEL_CONFIG
            or any(type(config.get(key)) is not type(value) for key, value in FROZEN_MODEL_CONFIG.items())):
        raise ValueError("checkpoint model_config differs from the frozen B1 configuration")
    for name, expected in (("epoch", 20), ("seed", 2026), ("checkpoint_format_version", 2)):
        if type(saved.get(name)) is not int or saved[name] != expected:
            raise ValueError(f"checkpoint {name} differs from the frozen B1 metadata")
    score = saved.get("best_valid_ndcg@20")
    if (type(score) not in (int, float) or not math.isfinite(score)
            or not math.isclose(score, FROZEN_REFERENCE_NDCG20, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("checkpoint validation-selected score differs")
    if not isinstance(saved.get("model"), dict) or not saved["model"]:
        raise ValueError("checkpoint has no model state")


def validation_batch(bundle: DatasetBundle, users: list[str]) -> tuple[list[list[int]], list[tuple[int, int]]]:
    """Model input is last 50; zero-based seen-mask columns use ALL history."""
    histories: list[list[int]] = []
    seen_pairs: list[tuple[int, int]] = []
    for row_index, user in enumerate(users):
        history = bundle.valid[user]["history"]
        visible = history[-50:]
        histories.append([0] * (50 - len(visible)) + visible)
        seen_pairs.extend((row_index, item - 1) for item in sorted(set(history)))
    return histories, seen_pairs


def reproduce(*, prepared_data: str | Path, bundle_root: str | Path,
              checkpoint: str | Path, model_source: str | Path,
              output: str | Path, device: str = "cuda", batch_size: int = 512) -> dict[str, Any]:
    started = time.perf_counter()
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("batch_size must be 1..4096")
    output = require_new_output(output)
    # Byte checks and public identity happen before importing/loading a model.
    pinned_bytes(prepared_data, FROZEN_DATA_SHA256, "prepared data")
    source_bytes = pinned_bytes(model_source, FROZEN_MODEL_SOURCE_SHA256, "reviewed SASRec source")
    checkpoint_bytes = pinned_bytes(checkpoint, FROZEN_CHECKPOINT_SHA256, "B1 checkpoint")
    bundle = DatasetBundle(bundle_root)  # Reads public files only.
    verify_public_bundle(bundle)

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no implicit CPU fallback")
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    verify_checkpoint_metadata(saved)
    del checkpoint_bytes
    # Execute only bytes matching the fixed reviewed module pin. This is not
    # dynamic candidate loading; there is no CLI option to replace the pin.
    frozen_module = types.ModuleType("reclab_reviewed_frozen_sasrec")
    exec(compile(source_bytes, "<reviewed-frozen-sasrec>", "exec"), frozen_module.__dict__)
    torch.manual_seed(2026)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if device == "cuda":
        torch.cuda.manual_seed_all(2026)
    model = frozen_module.SASRec(**FROZEN_MODEL_CONFIG)
    model.load_state_dict(saved["model"], strict=True)
    model = model.to(device).eval()
    users = sorted(bundle.valid)
    predictions: dict[str, list[int]] = {}
    if device == "cuda":
        torch.cuda.synchronize()
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(users), batch_size):
            batch_users = users[start:start + batch_size]
            histories, seen = validation_batch(bundle, batch_users)
            tensor = torch.tensor(histories, dtype=torch.long, device=device)
            scores = model.catalog_logits(model.encode(tensor))
            if seen:
                rows, columns = zip(*seen)
                scores[torch.tensor(rows, device=device), torch.tensor(columns, device=device)] = -torch.inf
            top = (scores.topk(20, dim=-1).indices + 1).cpu().tolist()
            predictions.update(zip(batch_users, top))
    if device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_started
    # Independent scorer validates exact users, unseen items, coverage and ranks.
    grade = bundle.grade(predictions, "valid")
    prediction_bytes = canonical_json(predictions).encode("utf-8")
    grade_bytes = canonical_json(grade).encode("utf-8")
    observed = grade["metrics"]["ndcg@20"]
    metadata = {
        "schema_version": 1, "purpose": "trusted_frozen_checkpoint_validation_reproduction",
        "candidate_executor": False, "trained": False, "split": "valid",
        "test_evaluated": False, "test_used_for_selection": False,
        "checkpoint_selected_before_this_run": True, "evaluation_sampled": False,
        "prepared_data_sha256": FROZEN_DATA_SHA256,
        "checkpoint_sha256": FROZEN_CHECKPOINT_SHA256,
        "model_source_sha256": FROZEN_MODEL_SOURCE_SHA256,
        "exporter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "public_data_hash": bundle.data_hash, "dataset_id": bundle.dataset_id,
        "model_config": FROZEN_MODEL_CONFIG, "checkpoint_epoch": 20, "checkpoint_seed": 2026,
        "reference_valid_ndcg@20": FROZEN_REFERENCE_NDCG20,
        "observed_minus_reference_ndcg@20": observed - FROZEN_REFERENCE_NDCG20,
        "reference_reproduced_at_1e-10": math.isclose(observed, FROZEN_REFERENCE_NDCG20, rel_tol=0, abs_tol=1e-10),
        "metrics": grade["metrics"], "protocol_hash": grade["protocol_hash"],
        "evaluation_hash": grade["evaluation_hash"],
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "grade_sha256": hashlib.sha256(grade_bytes).hexdigest(),
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "python_version": sys.version.split()[0], "platform": platform.platform(),
        "batch_size": batch_size, "model_visible_history": "last_50_left_padded",
        "seen_mask_history": "complete_training_history", "topk": 20,
        "tie_break": "torch.topk_original_B1_rule_no_additional_item_sort",
        "inference_seconds": inference_seconds, "total_seconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for relative, raw in {"predictions.json": prediction_bytes, "grade.json": grade_bytes,
                          "metadata.json": canonical_json(metadata).encode("utf-8")}.items():
        atomic_write(confined_path(output, relative), raw)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    result = reproduce(prepared_data=args.prepared_data, bundle_root=args.bundle,
                       checkpoint=args.checkpoint, model_source=args.model_source,
                       output=args.output, device=args.device, batch_size=args.batch_size)
    print(canonical_json(result), flush=True)
    if not result["reference_reproduced_at_1e-10"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
