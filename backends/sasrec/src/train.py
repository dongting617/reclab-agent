"""Configured next-item model training in a public-only experiment job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from data import canonical_json, load_config, load_public_data


class PrefixTargets:
    """Same prefix-to-next-item samples as the original SASRec trainer."""

    def __init__(self, train):
        self.rows = [(history[:position], history[position])
                     for history in train.values() for position in range(1, len(history))]
        if not self.rows:
            raise ValueError("training histories contain no prefix-to-next-item examples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index


def train(data_path, output, config_path):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from runtime import (build_model, digest, histories_tensor, implementation_hash, initialize,
                         model_config, predict_valid, save_checkpoint,
                         validation_ndcg20, write_json)

    data, config = load_public_data(data_path), load_config(config_path)
    device, seed = initialize(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("training output must be empty; existing results are never overwritten")
    examples = PrefixTargets(data["train"])

    def collate(indices):
        rows = [examples.rows[index] for index in indices]
        histories, targets = zip(*rows)
        return (histories_tensor(histories, config["max_length"], "cpu"),
                torch.tensor(targets, dtype=torch.long) - 1)

    loader = DataLoader(examples, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0,
                        collate_fn=collate, drop_last=False)
    selected_config = model_config(config, data["num_items"])
    model = build_model(config, data["num_items"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    loss_function = nn.CrossEntropyLoss()
    metadata = {"format": "reclab-next-item-public-v1", "data_hash": data["data_hash"],
                "dataset_id": data["manifest"]["dataset_id"], "config_hash": digest(config),
                "implementation_hash": implementation_hash(), "model_config": selected_config,
                "seed": seed, "config": config, "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "train_users": len(data["train"]), "train_examples": len(examples),
                "valid_users": len(data["valid"]), "test_evaluated": False,
                "selection_metric": "valid_ndcg@20", "official_score": "external_grader"}
    write_json(output / "training.json", metadata)
    best_score, selected_epoch, stale_epochs = -1.0, 0, 0
    for epoch in range(1, config["epochs"] + 1):
        started = time.monotonic()
        model.train()
        total_loss, count = 0.0, 0
        for histories, targets in loader:
            histories, targets = histories.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.catalog_logits(model.encode(histories))
            if logits.shape != (len(targets), data["num_items"]):
                raise ValueError("model scores must have shape [batch, catalog_items]")
            loss = loss_function(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss; checkpoint was not accepted")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(targets)
            count += len(targets)
        predictions = predict_valid(model, data, device, config["eval_batch_size"])
        score = validation_ndcg20(predictions, data["valid"])
        row = {"epoch": epoch, "train_loss": total_loss / count,
               "selection_valid_ndcg@20": score, "seconds": time.monotonic() - started}
        with (output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(row) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score:
            best_score, selected_epoch, stale_epochs = score, epoch, 0
            save_checkpoint(output / "best.pt", {**metadata, "model": model.state_dict(),
                                                  "epoch": epoch, "selection_valid_ndcg@20": score})
        else:
            stale_epochs += 1
            if config["patience"] and stale_epochs >= config["patience"]:
                break
    write_json(output / "training.json", {**metadata, "selected_epoch": selected_epoch,
                                          "selection_valid_ndcg@20": best_score})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="public data directory, e.g. /data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(args.data, args.output, args.config)


if __name__ == "__main__":
    main()
