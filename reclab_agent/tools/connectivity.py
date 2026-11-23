"""Model startup probe; never registered as an experiment capability."""
from reclab_agent.tools.contracts import ToolSpec

STARTUP_CHECK = ToolSpec("startup_check", "Confirm tool-call compatibility.", {
    "type": "object", "properties": {}, "required": [],
    "additionalProperties": False,
})

def startup_check_schema():
    return {"type": "function", "name": STARTUP_CHECK.name,
            "description": STARTUP_CHECK.description,
            "parameters": STARTUP_CHECK.input_schema, "strict": True}
