"""Research capability inventory, separate from the executable tool registry.

Planned entries describe integration work. They have no callable or input schema
and must never be passed to a model as executable function definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolCapability:
    name: str
    group: str
    availability: Literal["available", "planned"]
    responsibility: str


TOOL_CATALOG: tuple[ToolCapability, ...] = (
    ToolCapability("list_evidence", "evidence", "available", "List bundled evidence and its file identities."),
    ToolCapability("inspect_artifact", "evidence", "available", "Read artifact identity and JSON metadata."),
    ToolCapability("read_json_values", "evidence", "available", "Read explicit values with JSON pointers."),
    ToolCapability("compare_numeric_values", "evidence", "available", "Compute differences between finite recorded numbers."),
    ToolCapability("diagnose_beam_gap", "diagnostics", "available", "Analyze the bundled G2 Beam Search miss report."),
    ToolCapability("diagnose_frequency_gap", "diagnostics", "available", "Analyze bundled B1/G2 frequency-bucket gaps."),
    ToolCapability("search_knowledge", "knowledge", "planned", "Retrieve registered sources with locations and versions."),
    ToolCapability("search_source", "workspace", "planned", "Search the registered candidate source and configuration."),
    ToolCapability("read_source", "workspace", "planned", "Read registered source with content identity and location."),
    ToolCapability("apply_patch", "workspace", "planned", "Apply an approved, hash-checked edit and return its actual diff."),
    ToolCapability("validate_candidate", "workspace", "planned", "Check syntax and entrypoint contracts without host execution."),
    ToolCapability("submit_experiment", "experiments", "planned", "Submit an experiment inside a reviewed research plan."),
    ToolCapability("get_experiment", "experiments", "planned", "Read experiment state, bounded logs, artifacts and errors."),
    ToolCapability("cancel_experiment", "experiments", "planned", "Request cancellation and report observed stop confirmation."),
    ToolCapability("compare_experiments", "evaluation", "planned", "Compare independent reports with matching evaluation identities."),
)


def tool_catalog(*, availability: str | None = None) -> tuple[ToolCapability, ...]:
    """Return immutable inventory entries, optionally filtered by readiness."""
    if availability not in {None, "available", "planned"}:
        raise ValueError("availability must be available or planned")
    return tuple(entry for entry in TOOL_CATALOG if availability is None or entry.availability == availability)
