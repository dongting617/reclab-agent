"""Transactional text patches over explicitly registered candidate source files."""
from __future__ import annotations
import ast
import difflib
from pathlib import Path
from typing import Any
from reclab_agent.domain.contracts import content_hash, identifier
from reclab_agent.infrastructure.files import atomic_write, confined_path


class Workspace:
    """Register complete source inputs separately from the files an Agent may edit.

    With no explicit ``source_paths`` the original behavior is preserved: every
    registered file is editable. Read-only dependencies remain readable and are
    validated, hashed and copied into every snapshot.
    """

    ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml"}

    def __init__(self, root: str | Path, editable_paths: list[str], *,
                 source_paths: list[str] | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.editable_paths = frozenset(editable_paths)
        if not self.editable_paths:
            raise ValueError("explicit editable paths are required")
        self.source_paths = (self.editable_paths if source_paths is None
                             else frozenset(source_paths))
        if not self.source_paths or not self.editable_paths <= self.source_paths:
            raise ValueError("editable paths must be a subset of the registered source paths")
        for relative in self.source_paths:
            path = confined_path(self.root, relative)
            if (not relative.startswith(("src/", "configs/"))
                    or any(part in {"", ".", ".."} for part in relative.split("/"))
                    or path.suffix not in self.ALLOWED_EXTENSIONS):
                raise ValueError("only registered src/configs text files are allowed")

    def read(self, path: str) -> dict[str, str]:
        if path not in self.source_paths:
            raise ValueError("file is not in the candidate source allowlist")
        text = confined_path(self.root, path).read_text(encoding="utf-8")
        return {"path": path, "text": text, "sha256": content_hash(text)}

    def replace(self, path: str, old: str, new: str, expected_sha256: str) -> dict[str, Any]:
        """Apply an exact text edit; return the actual unified diff, not a claimed diff."""
        if path not in self.editable_paths:
            raise ValueError("file is not in the candidate editable allowlist")
        current = self.read(path)
        if current["sha256"] != expected_sha256:
            raise ValueError("stale patch: read current source before editing")
        if not isinstance(old, str) or not old or current["text"].count(old) != 1:
            raise ValueError("old text must match exactly once")
        if not isinstance(new, str) or len(new.encode("utf-8")) > 256_000:
            raise ValueError("replacement must be text of at most 256 KB")
        updated = current["text"].replace(old, new, 1)
        if len(updated.encode("utf-8")) > 1_000_000:
            raise ValueError("candidate source file exceeds 1 MB")
        diff = "".join(difflib.unified_diff(current["text"].splitlines(True), updated.splitlines(True),
                                           fromfile=path, tofile=path))
        if not diff:
            raise ValueError("patch has no effect")
        atomic_write(confined_path(self.root, path), updated.encode("utf-8"))
        return {"path": path, "before_sha256": current["sha256"], "after_sha256": content_hash(updated), "diff": diff}

    def validate(self) -> dict[str, Any]:
        errors = []
        for path in sorted(self.source_paths):
            try:
                source = self.read(path)["text"]
                if path.endswith(".py"):
                    ast.parse(source, filename=path)
                elif path.endswith(".json"):
                    import json
                    json.loads(source)
            except (ValueError, SyntaxError, OSError) as error:
                errors.append({"path": path, "error": str(error)})
        return {"ok": not errors, "errors": errors, "checks": "syntax_only; does not execute candidate code"}

    def snapshot(self, destination: str | Path) -> dict[str, Any]:
        files = {path: self.read(path)["text"] for path in sorted(self.source_paths)}
        source_hash = content_hash(files)
        base = Path(destination) / source_hash
        for relative, source in files.items():
            path = confined_path(base, relative)
            if path.exists() and path.read_text(encoding="utf-8") != source:
                raise ValueError("immutable snapshot was modified")
            if not path.exists():
                atomic_write(path, source.encode("utf-8"))
        return {"source_hash": source_hash, "directory": str(base), "files": sorted(files)}


def create_workspace(root: str | Path, name: str, files: dict[str, str], *,
                     editable_paths: list[str] | None = None) -> Workspace:
    directory = Path(root) / identifier(name)
    if directory.exists():
        raise ValueError("workspace already exists")
    source_paths = list(files)
    workspace = Workspace(directory, source_paths if editable_paths is None else editable_paths,
                          source_paths=source_paths)
    for relative, source in files.items():
        if not isinstance(source, str):
            raise ValueError("candidate source must be text")
        atomic_write(confined_path(directory, relative), source.encode("utf-8"))
    return workspace
