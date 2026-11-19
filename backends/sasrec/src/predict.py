"""Export validation rankings in a separate job from read-only trained weights."""
from __future__ import annotations

import argparse
from pathlib import Path

from data import load_config, load_public_data


def predict(data_path, model_path, output, config_path):
    from runtime import initialize, load_checkpoint, predict_valid, write_json

    data, config = load_public_data(data_path), load_config(config_path)
    device, seed = initialize(config)
    model_path, output = Path(model_path).resolve(), Path(output).resolve()
    if output == model_path or model_path in output.parents or output in model_path.parents:
        raise ValueError("prediction output must be separate from the read-only model directory")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("prediction output must be empty; existing results are never overwritten")
    model, selected = load_checkpoint(model_path / "best.pt", data, config, seed)
    predictions = predict_valid(model.to(device), data, device, config["eval_batch_size"])
    write_json(output / "predictions.json", predictions)
    write_json(output / "prediction.json", {"data_hash": data["data_hash"], "seed": seed,
                                           "selected_epoch": selected["epoch"], "split": "valid",
                                           "users": len(predictions), "test_evaluated": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), required=True)
    args = parser.parse_args()
    predict(args.data, args.model, args.output, args.config)


if __name__ == "__main__":
    main()
