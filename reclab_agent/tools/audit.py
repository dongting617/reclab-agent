"""Read-only, immutable evidence handoff. No access to the research toolbox."""
from __future__ import annotations
import copy
from reclab_agent.tools.contracts import ToolSpec

AUDIT_PROTOCOL_VERSION = 2


class AuditToolbox:
    """Only host-recorded tool observations are visible; paths grant no file access."""

    def __init__(self, packet):
        self._packet = copy.deepcopy(packet)

    @property
    def specs(self):
        return [ToolSpec("read_audit_evidence", "Read the frozen host evidence packet, including provenance and scope.",
                         {"type": "object", "properties": {}, "additionalProperties": False})]

    def invoke(self, name, arguments=None):
        if name != "read_audit_evidence":
            return {"ok": False, "tool": name, "error": {"code": "TOOL_NOT_ALLOWED", "message": "Audit is read-only."}}
        if arguments:
            return {"ok": False, "tool": name, "error": {"code": "INVALID_ARGUMENTS", "message": "No paths or arguments accepted."}}
        return {"ok": True, "tool": name, "result": copy.deepcopy(self._packet)}


AUDIT_SCHEMA = {
    "type": "function", "name": "submit_audit",
    "description": "Submit a claim audit as data. This grants no experiment permission.",
    "strict": True, "parameters": {
        "type": "object", "properties": {
            "verdict": {"type": "string", "enum": ["approved", "rejected", "insufficient_evidence"]},
            "findings": {"type": "array", "items": {"type": "object", "properties": {
                "message": {"type": "string"}, "evidence_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["message", "evidence_ids"], "additionalProperties": False}},
            "summary": {"type": "string"}},
        "required": ["verdict", "findings", "summary"], "additionalProperties": False}}


def validate_audit(value, evidence_ids):
    if (not isinstance(value, dict) or set(value) != {"verdict", "findings", "summary"}
            or not isinstance(value["verdict"], str)
            or value["verdict"] not in {"approved", "rejected", "insufficient_evidence"}
            or not isinstance(value["summary"], str) or not value["summary"].strip()
            or not isinstance(value["findings"], list)):
        raise ValueError("invalid audit schema")
    for finding in value["findings"]:
        if (not isinstance(finding, dict) or set(finding) != {"message", "evidence_ids"}
                or not isinstance(finding["message"], str) or not finding["message"].strip()
                or not isinstance(finding["evidence_ids"], list)
                or any(not isinstance(ref, str) or ref not in evidence_ids for ref in finding["evidence_ids"])):
            raise ValueError("invalid audit finding or evidence reference")
    if value["verdict"] != "insufficient_evidence" and not any(f["evidence_ids"] for f in value["findings"]):
        raise ValueError("audit verdict requires cited observations")
    return value
