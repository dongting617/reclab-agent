"""Workspace function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.tools.contracts import ToolError
from reclab_agent.tools.tool_group import ToolGroup

class WorkspaceTools(ToolGroup):
    definitions = (('read_source',
      'Read an approved candidate file and its current hash.',
      {'path': {'type': 'string'}}),
     ('apply_patch',
      'Apply one exact, hash-checked edit within the reviewed plan.',
      {'path': {'type': 'string'},
       'old': {'type': 'string'},
       'new': {'type': 'string'},
       'expected_sha256': {'type': 'string'}}),
     ('validate_candidate', 'Check candidate syntax without executing it.', {}))

    def read_source(self, path: str) -> dict[str, Any]:
        return self.workspace.read(path)


    def apply_patch(self, path: str, old: str, new: str, expected_sha256: str) -> dict[str, Any]:
        self._authorize()
        if self.active_stage is not None and self.active_stage != "experiment":
            raise ToolError("WRONG_STAGE", "source edits require the experiment stage")
        if path not in self.plan.editable_files:
            raise ToolError("OUTSIDE_PLAN", "file is outside the reviewed plan")
        if self._patches >= self.plan.budget.max_patches:
            raise ToolError("BUDGET_EXCEEDED", "patch attempt budget exhausted")
        self._patches += 1
        self.save_state()
        result = self.workspace.replace(path, old, new, expected_sha256)
        self.save_state()
        return result


    def validate_candidate(self) -> dict[str, Any]:
        return self.workspace.validate()
