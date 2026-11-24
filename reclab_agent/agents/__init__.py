"""Interchangeable agent strategies sharing the same execution core."""

from reclab_agent.agents.factory import AGENT_TYPES, create_agent
from reclab_agent.agents.plan_execute_agent import PlanExecuteAgent
from reclab_agent.agents.react_agent import ReActAgent
from reclab_agent.agents.reflection_agent import ReflectionAgent
from reclab_agent.agents.simple_agent import SimpleAgent
from reclab_agent.agents.research_audit_agent import ResearchAuditAgent

__all__ = [
    "ReActAgent", "SimpleAgent", "PlanExecuteAgent",
    "ReflectionAgent", "ResearchAuditAgent", "AGENT_TYPES", "create_agent",
]
