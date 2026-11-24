"""CLI adapters for project initialization, ledger inspection and knowledge."""
from __future__ import annotations
from pathlib import Path
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.infrastructure.files import atomic_write, read_regular_file
from reclab_agent.experiments.baselines import checkpoint_manifest
from reclab_agent.domain.configuration import load_task_configuration, read_json_strict
from reclab_agent.application.launch import prepare_launch
from reclab_agent.experiments.outcomes import scored_record
from reclab_agent.memory.project import Project, knowledge_scope


def project_command(args):
    config = load_task_configuration(args.config)
    _, plan, inspection = prepare_launch(config)
    project = Project(config.project_directory or config.output_directory / "project", retrieval=config.retrieval)
    scope = knowledge_scope(config, inspection)
    if args.project_operation == "init":
        path = project.root / "project.json"
        if path.exists():
            raise ValueError("project already initialized")
        snapshot = {"directory": str(config.source_directory), "files": inspection["source"]["files"],
                    "source_hash": content_hash({p: (config.source_directory / p).read_text(encoding="utf-8")
                                                 for p in inspection["source"]["files"]})}
        commit = project.commit_source(snapshot, "initial")
        metadata = {"schema_version": 1, "task_id": config.task.task_id, "source": str(config.source_directory),
                    "data": str(config.dataset_directory), "scope": scope, "initial_commit": commit,
                    "seed": config.seeds[0], "evaluation": "valid"}
        atomic_write(path, canonical_json(metadata).encode("utf-8"))
        return {"project": str(project.root), **metadata}
    if args.project_operation == "experiments":
        entries = project.experiments(scope)
        if args.experiment_id:
            entries = [x for x in entries if x["id"] == args.experiment_id]
        return {"experiments": entries}
    if args.project_operation == "runs":
        with project.connect() as db:
            rows = db.execute("SELECT id,payload FROM runs ORDER BY id").fetchall()
        import json
        return {"runs": [{"id": x[0], "state": json.loads(x[1])} for x in rows]}
    if args.project_operation == "search":
        return {"scope": scope, "matches": project.search(scope, args.query)}
    if args.project_operation == "reindex":
        return {"scope": scope, "chunks": len(project.reindex(scope).chunks)}
    if args.project_operation == "note":
        text = read_regular_file(Path(args.input), max_bytes=256_000).decode("utf-8")
        return project.note("experience", scope, args.note_id, text, replaces=args.replaces)
    saved = read_json_strict(args.input, max_bytes=32_000_000)
    imported = []
    for row in saved.get("experiments", []):
        record = row.get("record", row)
        try:
            entry = scored_record(Path(args.input).absolute().parent, record, plan)
            manifest = checkpoint_manifest(Path(args.input).absolute().parent, entry)
            source = Path(args.input).absolute().parent / "snapshots" / record["source_hash"]
            files = {p.relative_to(source).as_posix(): read_regular_file(p, max_bytes=1_000_000).decode("utf-8").replace("\r\n", "\n")
                     for p in source.rglob("*") if p.is_file()} if source.is_dir() else {}
            verified = ("metrics" in entry and any(p.endswith((".pt", ".pth")) for p in manifest)
                        and content_hash(files) == record["source_hash"]
                        and saved.get("baseline_identity") is not None)
        except (OSError, ValueError, KeyError):
            entry, verified = {"record": record, "verification": "unverified"}, False
        project.record(entry, scope, Path(args.input).absolute().parent, verified=verified)
        project.note("experience", scope, "import-" + record["experiment_id"],
                     "# Imported experiment\n\nReusable verification: " + str(verified) + "\n\nEvidence root: "
                     + str(Path(args.input).absolute().parent) + "\n\n```json\n"
                     + canonical_json(entry) + "\n```\n")
        imported.append({"experiment_id": record["experiment_id"], "verified": verified})
    return {"imported": imported}
