"""Data-only role output schemas; these functions never dispatch business actions."""
from reclab_agent.tools.contracts import ToolSpec
from copy import deepcopy

ROLE_OUTPUTS = {
    "plan": ToolSpec("role_plan", "Submit a bounded plan as data; this grants no execution permission.", {
        "type": "object", "properties": {"steps": {"type": "array", "items": {"type": "string"},
                                                  "minItems": 1, "maxItems": 5}},
        "required": ["steps"], "additionalProperties": False}),
    "review": ToolSpec("role_review", "Submit an evidence review as data; this does not authorize experiments.", {
        "type": "object", "properties": {"approved": {"type": "boolean"}, "feedback": {"type": "string"}},
        "required": ["approved", "feedback"], "additionalProperties": False}),
}


def role_output_schema(phase):
    spec = ROLE_OUTPUTS[phase]
    return {"type": "function", "name": spec.name, "description": spec.description,
            "parameters": deepcopy(spec.input_schema), "strict": True}
