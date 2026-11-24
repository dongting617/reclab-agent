"""Read-only qualification of research progress; no training or grading side effects."""
from __future__ import annotations


def prepare_entries(entries, report, mode):
    """Bind process/audit requirements to candidates before any best selection."""
    hypotheses = {h.get("candidate_id"): h for h in report.get("hypotheses", []) if isinstance(h, dict)}
    audits = report.get("audits", {})
    audits = audits if isinstance(audits, dict) else {}
    for entry in entries:
        candidate = entry["record"].get("candidate_id")
        hypothesis = hypotheses.get(candidate, {})
        definitions = hypothesis.get("process_metrics") or []
        process = entry.get("process_metrics")
        process = process if isinstance(process, dict) else {}
        observed = process.get("metrics") or []
        required = {d.get("name") for d in definitions if isinstance(d, dict) and d.get("purpose") == "required"}
        required.update(d.get("name") for d in observed if isinstance(d, dict) and d.get("purpose") == "required")
        passes = {d.get("name") for d in observed if isinstance(d, dict)
                  and d.get("purpose") == "required" and d.get("passes") is True and d.get("status") == "observed"}
        entry["process_pass"] = (entry["eligible"] is True and process.get("passes_required", True) is True and (not required or (
            entry.get("eligibility_recorded") and process.get("passes_required") is True
            and bool(process.get("definition_hash")) and required <= passes)))
        audit = audits.get(candidate, entry.get("audit"))
        entry["audit"] = audit if isinstance(audit, dict) else None
        entry["audit_required"] = mode == "research_audit"
        assessment = (entry["audit"] or {}).get("audit")
        assessment = assessment if isinstance(assessment, dict) else {}
        entry["audit_pass"] = not entry["audit_required"] or bool(entry["audit"]
            and entry["audit"].get("status") == "completed" and entry["audit"].get("audit_approved") is True
            and assessment.get("verdict") == "approved")
        entry["stop_confirmed"] = report.get("requires_reconciliation") is not True
        entry["hypothesis"] = hypothesis


def progress_view(groups, baseline, primary):
    """Keep strategy groups separate: their file order is not a global chronology."""
    return {"primary": primary, "scope": "validation_single_seed", "groups": [
        {"key": group["key"], "name": group["name"],
         "baseline": {"id": baseline["id"], "experiment_id": baseline["experiment_id"],
                      "value": baseline["metrics"][primary] if baseline["verified"] and baseline["metrics"] else None},
         "points": [{"experiment_id": entry["experiment_id"], "id": entry["id"],
                     "order": index + 1, "value": entry["plot_value"], "best": entry["best_value"],
                     "milestone": entry["milestone"], "previous_delta": entry["previous_delta"],
                     "delta": entry["deltas"].get(primary), "relative": entry["relative"].get(primary),
                     "qualified": entry["comparison_eligible"], "decision": entry["decision"]}
                    for index, entry in enumerate(group["entries"])]}
        for group in groups]}
