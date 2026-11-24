"""Read-only process evidence evaluation; candidate code is never imported here."""
from __future__ import annotations
import math
import operator
import hashlib
import json
from reclab_agent.domain.contracts import content_hash
from reclab_agent.infrastructure.files import confined_path, read_regular_file
from reclab_agent.domain.configuration import read_json_strict


OPERATORS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
             "==": operator.eq, "!=": operator.ne}


def validate_definitions(definitions):
    names = set()
    for row in definitions:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"]
                or row["name"] in names or row.get("purpose") not in {"diagnostic", "required"}
                or not isinstance(row.get("expected"), str)):
            raise ValueError("process metrics require unique names, purpose and expected observation")
        names.add(row["name"])
        if row["purpose"] == "required" and (row.get("operator") not in OPERATORS
                or type(row.get("threshold")) not in (float, int) or not math.isfinite(row["threshold"])):
            raise ValueError("required process metrics need an operator and finite threshold")


def evaluate_process_metrics(output, record, definitions):
    validate_definitions(definitions)
    path = confined_path(output, "experiments/" + record["experiment_id"] + "/predict/process_metrics.json")
    evidence = read_json_strict(path, max_bytes=1_000_000) if path.exists() else {}
    values = evidence.get("metrics", {})
    if not isinstance(values, dict) or any(type(v) not in (float, int) or not math.isfinite(v) for v in values.values()):
        raise ValueError("process evidence must contain finite numeric values")
    values = dict(values)
    derived = []
    training_names = {"training_epochs_reached", "training_loss_final"}
    requested = {row["name"] for row in definitions} & training_names
    definition_hash = evidence.get("definition_hash")
    if requested:
        definition_hash = content_hash({"worker": definition_hash, "training_log_summary_version": 1})
        log = confined_path(output, "experiments/" + record["experiment_id"] + "/train/epochs.jsonl")
        if log.exists():
            raw = read_regular_file(log, max_bytes=4_000_000)
            try:
                rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
                valid = bool(rows) and all(type(row.get("epoch")) is int and row["epoch"] == i
                    and type(row.get("train_loss")) in (int, float) and math.isfinite(row["train_loss"])
                    for i, row in enumerate(rows, 1))
            except (ValueError, AttributeError, UnicodeError):
                valid = False
            if not valid:
                raise ValueError("training log summary requires contiguous epochs and finite losses")
            summary = {"training_epochs_reached": len(rows), "training_loss_final": rows[-1]["train_loss"]}
            for name in requested - values.keys():
                values[name] = summary[name]
            derived.append({"source": str(log), "sha256": hashlib.sha256(raw).hexdigest(),
                            "collector": "training_log_summary_v1", "metrics": sorted(requested)})
    results = []
    for definition in definitions:
        value = values.get(definition["name"])
        matched = (OPERATORS[definition["operator"]](value, definition["threshold"])
                   if value is not None and definition.get("operator") in OPERATORS else None)
        results.append({**definition, "value": value, "passes": matched,
                        "status": "missing" if value is None else "observed"})
    return {"definition_hash": definition_hash, "metrics": results, "derived_evidence": derived,
            "source": str(path) if evidence else None,
            "passes_required": all(x["passes"] is True for x in results if x["purpose"] == "required")}
