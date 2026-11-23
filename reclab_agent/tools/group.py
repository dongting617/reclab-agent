"""A capability group shares one study context with other groups."""
from reclab_agent.tools.contracts import ToolSpec

class ToolGroup:
    definitions = ()

    def __init__(self, context):
        object.__setattr__(self, "context", context)
        for name, _, _ in self.definitions:
            context.operations[name] = getattr(self, name)

    def __getattr__(self, name):
        context = object.__getattribute__(self, "context")
        if name in context.operations:
            return context.operations[name]
        return getattr(context, name)

    def __setattr__(self, name, value):
        setattr(self.context, name, value)

    def register(self, registry, names=None):
        for name, description, properties in self.definitions:
            if names is None or name in names:
                registry.register_function(ToolSpec(name, description, {
                    "type": "object", "properties": properties,
                    "required": list(properties), "additionalProperties": False,
                }), getattr(self, name))
