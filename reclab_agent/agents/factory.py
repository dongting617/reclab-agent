"""Select an execution strategy without coupling it to model experiments."""
from __future__ import annotations
from reclab_agent.agents.plan_execute_agent import PlanExecuteAgent
from reclab_agent.agents.react_agent import ReActAgent
from reclab_agent.agents.reflection_agent import ReflectionAgent
from reclab_agent.agents.simple_agent import SimpleAgent


AGENT_TYPES = {
    "react": ReActAgent,
    "single": ReActAgent,
    "simple": SimpleAgent,
    "plan_execute": PlanExecuteAgent,
    "reflection": ReflectionAgent,
    "reviewer": ReflectionAgent,
}


def create_agent(kind: str, toolbox, backend, **kwargs):
    """Build an agent; legacy TaskSpec mode names remain valid aliases."""
    try:
        agent_type = AGENT_TYPES[kind]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown agent type: {kind}") from error
    return agent_type(toolbox, backend, **kwargs)
