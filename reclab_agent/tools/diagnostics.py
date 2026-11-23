"""Diagnostics function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.tools.contracts import ToolError
from reclab_agent.tools.group import ToolGroup

class DiagnosticTools(ToolGroup):
    definitions = (('write_diagnostic',
      'Write evaluate(context) diagnostic code under src/diagnostics; executed only by the '
      'Worker.',
      {'path': {'type': 'string'}, 'text': {'type': 'string'}}),
     ('run_diagnostics',
      'Read versioned Worker process metrics and evaluate the registered conditions.',
      {'experiment_id': {'type': 'string'}}))

    def write_diagnostic(self, path: str, text: str) -> dict[str, Any]:
        import ast
        from reclab_agent.infrastructure.files import atomic_write
        from reclab_agent.infrastructure.files import confined_path
        from reclab_agent.domain.contracts import content_hash
        self._authorize()
        if self.active_stage != "experiment" or not self.hypothesis:
            raise ToolError("WRONG_STAGE", "diagnostic edits require a registered experiment hypothesis")
        if not path.startswith("src/diagnostics/") or not path.endswith(".py"):
            raise ValueError("diagnostic code must be under src/diagnostics with a .py suffix")
        if not isinstance(text, str) or len(text.encode("utf-8")) > 128_000:
            raise ValueError("diagnostic source must be bounded text")
        ast.parse(text, filename=path)
        if self._patches >= self.plan.budget.max_patches:
            raise ToolError("BUDGET_EXCEEDED", "patch budget exhausted")
        self._patches += 1
        self.save_state()
        atomic_write(confined_path(self.workspace.root, path), text.encode("utf-8"))
        self.workspace.source_paths = self.workspace.source_paths | {path}
        self.save_state()
        return {"path": path, "sha256": content_hash(text), "execution": "Worker prediction phase only"}


    def run_diagnostics(self, experiment_id: str) -> dict[str, Any]:
        from reclab_agent.experiments.diagnostics import evaluate_process_metrics
        record = self.get_experiment(experiment_id)
        return evaluate_process_metrics(self.snapshots_root.parent, record,
                                        (self.hypothesis or {}).get("process_metrics", []))
