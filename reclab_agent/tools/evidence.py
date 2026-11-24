"""Read-only tools over registered recommendation evidence."""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable
from reclab_agent.tools.contracts import ToolError, ToolSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_pointer_token(token: str) -> str:
    if re.search(r"~(?![01])", token):
        raise ToolError("INVALID_POINTER", "JSON pointer escapes must be '~0' or '~1'")
    return token.replace("~1", "/").replace("~0", "~")


def json_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer without silently coercing types."""
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ToolError("INVALID_POINTER", "JSON pointer must be empty or start with '/'")
    current = document
    for raw_token in pointer[1:].split("/"):
        token = _decode_pointer_token(raw_token)
        if isinstance(current, dict):
            if token not in current:
                raise ToolError("POINTER_NOT_FOUND", f"Object key not found: {token}")
            current = current[token]
        elif isinstance(current, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ToolError("INVALID_POINTER", f"Invalid array index: {token}")
            index = int(token)
            if index < 0 or index >= len(current):
                raise ToolError("POINTER_NOT_FOUND", f"List index out of range: {index}")
            current = current[index]
        else:
            raise ToolError("POINTER_NOT_FOUND", f"Cannot descend through scalar at: {token}")
    return current


class EvidenceToolbox:
    """Read and compare confined evidence artifacts, including bundled diagnostics."""

    GAP_REPORT = "evidence/g2-h128-gap-vs-b1-full-valid-20261007.json"
    FREQUENCY_AUDIT = "evidence/frequency-ce-completion-audit-20261008.json"
    ENSEMBLE_REPORT = "evidence/msp-ensemble-confirmation-parsed-20261009.json"

    def __init__(self, root: str | Path):
        # pathlib.resolve() calls GetFinalPathNameByHandle on Windows and can
        # fail on otherwise readable non-ASCII workspace paths. abspath still
        # normalizes '..' and keeps the sandbox check deterministic.
        self.root = Path(os.path.abspath(os.fspath(root)))
        if not self.root.is_dir():
            raise ToolError("ROOT_NOT_FOUND", f"RecLab root does not exist: {self.root}")
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {
            "list_evidence": self.list_evidence,
            "inspect_artifact": self.inspect_artifact,
            "read_json_values": self.read_json_values,
            "compare_numeric_values": self.compare_numeric_values,
            "diagnose_beam_gap": self.diagnose_beam_gap,
            "diagnose_frequency_gap": self.diagnose_frequency_gap,
        }

    @property
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("list_evidence", "List committed diagnostic evidence without reading model weights.", {
                "type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}},
                "additionalProperties": False,
            }),
            ToolSpec("inspect_artifact", "Inspect file identity, size and top-level JSON metadata.", {
                "type": "object", "required": ["path"], "properties": {
                    "path": {"type": "string"}, "expected_sha256": {"type": "string"}},
                "additionalProperties": False,
            }),
            ToolSpec("read_json_values", "Read explicit JSON pointers from one evidence artifact.", {
                "type": "object", "required": ["path", "pointers"], "properties": {
                    "path": {"type": "string"}, "pointers": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
                "additionalProperties": False,
            }),
            ToolSpec("compare_numeric_values", "Compare two numeric JSON values and report absolute and relative change.", {
                "type": "object", "required": ["left_path", "left_pointer", "right_path", "right_pointer"],
                "properties": {name: {"type": "string"} for name in
                    ("left_path", "left_pointer", "right_path", "right_pointer")},
                "additionalProperties": False,
            }),
            ToolSpec("diagnose_beam_gap", "Determine whether constrained Beam Search is the main G2 miss source.", {
                "type": "object", "properties": {}, "additionalProperties": False,
            }),
            ToolSpec("diagnose_frequency_gap", "Measure B1 versus G2 quality gaps across target-frequency buckets.", {
                "type": "object", "properties": {}, "additionalProperties": False,
            }),
        ]

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = arguments or {}
        if name not in self._tools:
            return {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"}}
        try:
            result = self._tools[name](**arguments)
            return {"ok": True, "tool": name, "result": result}
        except ToolError as error:
            return {"ok": False, "tool": name, "error": {"code": error.code, "message": str(error)}}
        except TypeError as error:
            return {"ok": False, "tool": name, "error": {"code": "INVALID_ARGUMENTS", "message": str(error)}}

    def _resolve(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ToolError("INVALID_PATH", "Path must be a non-empty string")
        candidate = Path(os.path.abspath(os.fspath(self.root / relative_path)))
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ToolError("PATH_OUTSIDE_ROOT", "Artifact path escapes the RecLab root") from error
        if not candidate.is_file():
            raise ToolError("ARTIFACT_NOT_FOUND", f"Artifact does not exist: {relative_path}")
        return candidate

    def _load_json(self, relative_path: str) -> tuple[Path, Any]:
        path = self._resolve(relative_path)
        if path.suffix.lower() != ".json":
            raise ToolError("NOT_JSON", f"Artifact is not a JSON file: {relative_path}")
        try:
            return path, json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ToolError("INVALID_JSON", f"Artifact cannot be decoded as JSON: {relative_path}") from error

    def list_evidence(self, limit: int = 100) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ToolError("INVALID_LIMIT", "limit must be an integer in [1, 200]")
        base = self.root / "evidence"
        if not base.is_dir():
            raise ToolError("EVIDENCE_DIR_NOT_FOUND", "evidence is missing")
        artifacts = []
        for path in sorted(base.glob("*.json"))[:limit]:
            relative = path.relative_to(self.root).as_posix()
            artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
        return {"count": len(artifacts), "truncated": len(list(base.glob("*.json"))) > limit,
                "artifacts": artifacts}

    def inspect_artifact(self, path: str, expected_sha256: str | None = None) -> dict[str, Any]:
        resolved = self._resolve(path)
        digest = _sha256(resolved)
        result: dict[str, Any] = {
            "path": resolved.relative_to(self.root).as_posix(), "bytes": resolved.stat().st_size,
            "sha256": digest, "suffix": resolved.suffix.lower(),
        }
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ToolError("INVALID_SHA256", "expected_sha256 must contain 64 hexadecimal characters")
            try:
                int(expected_sha256, 16)
            except ValueError as error:
                raise ToolError("INVALID_SHA256", "expected_sha256 must contain 64 hexadecimal characters") from error
            result["sha256_matches"] = digest == expected_sha256.lower()
        if resolved.suffix.lower() == ".json":
            _, document = self._load_json(path)
            result["json_type"] = type(document).__name__
            if isinstance(document, dict):
                result["top_level_keys"] = sorted(document)
                for key in ("schema_version", "experiment", "status", "split", "users", "date"):
                    if key in document:
                        result[key] = document[key]
        return result

    def read_json_values(self, path: str, pointers: list[str]) -> dict[str, Any]:
        if not isinstance(pointers, list) or not pointers or not all(isinstance(p, str) for p in pointers):
            raise ToolError("INVALID_POINTERS", "pointers must be a non-empty list of strings")
        if len(pointers) > 50:
            raise ToolError("TOO_MANY_POINTERS", "At most 50 JSON pointers may be read per call")
        resolved, document = self._load_json(path)
        return {"path": resolved.relative_to(self.root).as_posix(),
                "sha256": _sha256(resolved),
                "values": {pointer: json_pointer(document, pointer) for pointer in pointers}}

    def compare_numeric_values(self, left_path: str, left_pointer: str,
                               right_path: str, right_pointer: str) -> dict[str, Any]:
        left_file, left_doc = self._load_json(left_path)
        right_file, right_doc = self._load_json(right_path)
        left = json_pointer(left_doc, left_pointer)
        right = json_pointer(right_doc, right_pointer)
        if isinstance(left, bool) or not isinstance(left, (int, float)) or not math.isfinite(left):
            raise ToolError("NON_NUMERIC_VALUE", "left JSON value must be a finite number")
        if isinstance(right, bool) or not isinstance(right, (int, float)) or not math.isfinite(right):
            raise ToolError("NON_NUMERIC_VALUE", "right JSON value must be a finite number")
        delta = float(right) - float(left)
        relative = None if left == 0 else delta / abs(float(left))
        return {
            "left": {"path": left_file.relative_to(self.root).as_posix(), "pointer": left_pointer, "value": left},
            "right": {"path": right_file.relative_to(self.root).as_posix(), "pointer": right_pointer, "value": right},
            "right_minus_left": delta, "relative_change": relative,
        }

    def diagnose_beam_gap(self) -> dict[str, Any]:
        path, document = self._load_json(self.GAP_REPORT)
        bins = document["g2_missed_target_minus_returned_20th_bins"]
        suboptimal = int(bins["target_above_returned_20th"])
        scored_below = int(bins["target_below_returned_20th"])
        within = int(bins["within_tolerance"])
        underfilled = int(bins["underfilled"])
        misses = suboptimal + scored_below + within + underfilled
        if misses <= 0:
            raise ToolError("EMPTY_DIAGNOSTIC", "Beam diagnostic contains no misses")
        fraction = suboptimal / misses
        main_bottleneck = fraction >= 0.1
        return {
            "evidence_path": path.relative_to(self.root).as_posix(), "misses": misses,
            "search_suboptimal_misses": suboptimal, "search_suboptimal_fraction": fraction,
            "beam_width": document["beam_width"], "beam_is_main_bottleneck": main_bottleneck,
            "conclusion_code": "SEARCH_BOTTLENECK" if main_bottleneck else "SCORING_NOT_SEARCH",
        }

    def diagnose_frequency_gap(self) -> dict[str, Any]:
        path, document = self._load_json(self.GAP_REPORT)
        groups = document["groups"]["frequency_bucket"]
        rows = []
        for bucket in ("1-4", "5-19", "20-99", "100+"):
            row = groups[bucket]
            rows.append({
                "bucket": bucket, "users": row["users"],
                "b1_recall@20": row["b1_recall@20"], "g2_recall@20": row["g2_recall@20"],
                "recall_gap_b1_minus_g2": row["b1_recall@20"] - row["g2_recall@20"],
                "b1_ndcg@20": row["b1_ndcg@20"], "g2_ndcg@20": row["g2_ndcg@20"],
                "ndcg_gap_b1_minus_g2": row["b1_ndcg@20"] - row["g2_ndcg@20"],
            })
        worst = max(rows, key=lambda row: row["ndcg_gap_b1_minus_g2"])
        low_users = sum(row["users"] for row in rows[:2])
        return {
            "evidence_path": path.relative_to(self.root).as_posix(), "buckets": rows,
            "largest_b1_advantage_bucket": worst["bucket"],
            "largest_ndcg_gap_b1_minus_g2": worst["ndcg_gap_b1_minus_g2"],
            "low_frequency_users": low_users,
            "conclusion_code": "LOW_FREQUENCY_GAP_CONFIRMED" if all(
                row["ndcg_gap_b1_minus_g2"] > 0 for row in rows[:2]) else "LOW_FREQUENCY_GAP_NOT_CONFIRMED",
        }

