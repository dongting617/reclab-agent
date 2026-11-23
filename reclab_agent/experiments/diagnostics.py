"""Read-only process evidence evaluation; candidate code is never imported here."""
from __future__ import annotations
import math
import operator
from reclab_agent.domain.contracts import content_hash
from reclab_agent.infrastructure.files import confined_path
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
    results = []
    for definition in definitions:
        value = values.get(definition["name"])
        matched = (OPERATORS[definition["operator"]](value, definition["threshold"])
                   if value is not None and definition.get("operator") in OPERATORS else None)
        results.append({**definition, "value": value, "passes": matched,
                        "status": "missing" if value is None else "observed"})
    return {"definition_hash": evidence.get("definition_hash"), "metrics": results,
            "source": str(path) if evidence else None,
            "passes_required": all(x["passes"] is True for x in results if x["purpose"] == "required")}
