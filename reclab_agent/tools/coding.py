"""Small coding-evaluation tools over the existing confined workspace."""
from reclab_agent.tools.contracts import ToolError, ToolSpec
from reclab_agent.tools.registry import ToolRegistry
from reclab_agent.tools.workspace import WorkspaceTools


class CodingToolbox:
    def __init__(self, workspace, check, *, max_patches=4, max_checks=3):
        self.workspace, self.check = workspace, check
        self.max_patches, self.max_checks = max_patches, max_checks
        self.patches = self.checks = 0
        self.registry = ToolRegistry()
        handlers = {"read_source": workspace.read, "apply_patch": self.apply_patch,
                    "validate_candidate": workspace.validate}
        for name, description, properties in WorkspaceTools.definitions:
            self.registry.register_function(ToolSpec(name, description, {
                "type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}), handlers[name])
        self.registry.register_function(ToolSpec("run_checks", "Run public checks in the fixed CPU container; hidden answers are unavailable.",
            {"type": "object", "properties": {}, "required": [], "additionalProperties": False}), self.run_checks)

    @property
    def specs(self):
        return self.registry.specs

    def invoke(self, name, arguments=None):
        return self.registry.invoke(name, arguments)

    def apply_patch(self, path, old, new, expected_sha256):
        if self.patches >= self.max_patches:
            raise ToolError("BUDGET_EXCEEDED", "coding patch allowance exhausted")
        self.patches += 1
        return self.workspace.replace(path, old, new, expected_sha256)

    def run_checks(self):
        if self.checks >= self.max_checks:
            raise ToolError("BUDGET_EXCEEDED", "public check allowance exhausted")
        self.checks += 1
        return self.check()
