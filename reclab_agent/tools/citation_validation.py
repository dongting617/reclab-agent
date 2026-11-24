"""Validate structured JSON citations against captured observations, never live files."""
from __future__ import annotations
from pathlib import PurePosixPath, PureWindowsPath
from reclab_agent.core.llm import parse_json_response
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.tools.evidence import json_pointer


def check_claim_citations(text, observations):
    """Check location, not scientific support or hidden grading labels.

    Free-text answers are outside this structured citation contract. A JSON answer
    with citations must cite values actually returned by read_json_values.
    """
    try:
        claim = parse_json_response(text)
    except (ValueError, TypeError):
        return {"status": "not_applicable", "errors": []}
    if not isinstance(claim, dict) or "citations" not in claim:
        return {"status": "not_applicable", "errors": []}
    citations = claim["citations"]
    if not isinstance(citations, list) or not citations:
        return {"status": "failed", "errors": [{"code": "INVALID_CITATIONS", "message": "citations must be a nonempty list"}]}
    errors = []
    for index, ref in enumerate(citations):
        try:
            if (not isinstance(ref, dict) or not isinstance(ref.get("path"), str)
                    or not isinstance(ref.get("pointer"), str)):
                raise ValueError("Citation requires string path and pointer")
            path, pointer = ref["path"], ref["pointer"]
            if (not path or "\\" in path or "\x00" in path or PureWindowsPath(path).drive
                    or PurePosixPath(path).is_absolute() or any(p in {".", "..", ""} for p in path.split("/"))):
                raise ValueError("Citation path must be an observed relative path")
            matches = []
            for observation in observations:
                envelope = observation.get("result", {})
                result = envelope.get("result", {})
                if observation.get("tool") != "read_json_values" or not envelope.get("ok") or result.get("path") != path:
                    continue
                for base, value in result.get("values", {}).items():
                    if pointer == base or base == "" or pointer.startswith(base + "/"):
                        try:
                            matches.append(json_pointer(value, pointer[len(base):]))
                        except ValueError:
                            pass
            if not matches:
                raise ValueError("Pointer is invalid or was not observed; root is the empty string, '/' is an empty object key")
            if len({canonical_json(value) for value in matches}) != 1:
                raise ValueError("Citation is ambiguous: observations contain conflicting values")
        except (ValueError, TypeError, KeyError) as error:
            errors.append({"index": index, "code": "INVALID_CITATION", "message": str(error)})
    return {"status": "failed" if errors else "passed", "errors": errors}
