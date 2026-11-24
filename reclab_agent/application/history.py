"""Evidence-backed project archival and historical baseline selection."""
from __future__ import annotations
import json
from pathlib import Path
from reclab_agent.domain.contracts import canonical_json
from reclab_agent.execution.worker import InputTree
from reclab_agent.infrastructure.files import atomic_write
from reclab_agent.experiments.baselines import baseline_identity, checkpoint_manifest
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.experiments.outcomes import scored_record


def archive_run(project, scope, config, plan, inspection, research, report):
    """Archive measured outcomes even when final formatting or interpretation fails."""
    output = config.output_directory
    for value in research.experiments:
        if any(x.get("reused_from") and x["record"]["experiment_id"] == value["experiment_id"]
               for x in report.get("experiments", [])):
            continue
        try:
            entry = scored_record(output, value, plan)
        except (OSError, ValueError, KeyError):
            entry = {"record": value, "verification": "unverified"}
        if config.task.mode == "research_audit" and value["candidate_id"] != plan.baseline_id:
            entry["audit"] = report.get("audits", {}).get(value["candidate_id"], {"status": "not_completed"})
        entry["git_commit"] = research.git_commits.get(value["experiment_id"])
        manifest = checkpoint_manifest(output, entry)
        verified = "metrics" in entry and bool(manifest) and any(p.endswith((".pt", ".pth")) for p in manifest)
        project.record(entry, scope, output, verified=verified)
        note = (f"# Experiment {value['experiment_id']}\n\nScope: {scope}\n"
                f"Evidence root: {output}\nVerification: {'verified' if verified else 'partial'}\n\n"
                "## Observed outcome\n\n```json\n" + json.dumps(entry, ensure_ascii=False, indent=2) + "\n```\n\n"
                "## Hypotheses and interpretations (not measured facts)\n\n```json\n"
                + json.dumps(report.get("hypotheses", []), ensure_ascii=False, indent=2) + "\n```\n\n"
                "## Decision\n\n```json\n" + json.dumps([x for x in report.get("comparison", [])
                    if x["experiment_id"] == value["experiment_id"]], ensure_ascii=False, indent=2) + "\n```\n")
        project.note("experience", scope, "exp-" + value["experiment_id"], note[:250_000])
        project.note("experiments", scope, "report-" + value["experiment_id"], note[:250_000])
        if not verified:
            continue
        snapshot_root = research.snapshots_root / value["source_hash"]
        identity = baseline_identity(config, inspection)
        identity["source"] = InputTree.freeze(snapshot_root, "source").sha256
        export = {"baseline_id": value["candidate_id"], "baseline_identity": identity,
                  "baseline_artifacts": manifest, "experiments": [entry]}
        atomic_write(output / ("baseline-" + value["experiment_id"] + ".json"), canonical_json(export).encode("utf-8"))
        if value["experiment_id"] == report.get("best_experiment_id"):
            project.accept(scope, value["experiment_id"])
    project.save_run(output.name, report, "report")


def select_project_baseline(project, scope, config, inspection, *, registry=None, output_directory=None):
    """Use the explicit accepted pointer, then an exact original baseline; never rank unreviewed scores."""
    if config.baseline_report:
        return config
    from dataclasses import replace
    rows = []
    current = project.current(scope)
    if current:
        rows.append(current)
    rows.extend(x for x in project.experiments(scope)
                if x["entry"]["record"]["candidate_id"] == config.task.baseline_id)
    for row in rows:
        if not row["verified"]:
            continue
        record = row["entry"]["record"]
        root = Path(row["root"])
        export = root / ("baseline-" + record["experiment_id"] + ".json")
        source = root / "snapshots" / record["source_hash"]
        if not export.is_file() or not source.is_dir():
            continue
        # The import tool does the full artifact check before any paid execution.
        selected = replace(config, source_directory=source, baseline_report=export)
        from reclab_agent.application.inspection import inspect_task
        check_config = replace(selected, output_directory=output_directory or config.output_directory / "sessions/baseline-inspection")
        candidate_inspection = inspect_task(check_config, registry=registry)
        if candidate_inspection["inputs_valid"]:
            expected = baseline_identity(selected, candidate_inspection)
            if read_json_strict(export).get("baseline_identity") == expected:
                return selected
    return config
