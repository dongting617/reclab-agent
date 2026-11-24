"""RecLab Agent: a verifiable tool environment for recommendation experiments."""

from reclab_agent.evaluation.benchmark import load_tasks, run_reference_suite
from reclab_agent.evaluation.traces import load_trace, replay_trace
from reclab_agent.core import Agent, AgentConfig, ModelBackend, ModelTurn, OpenAIResponsesBackend
from reclab_agent.agents import (
    ReActAgent,
    SimpleAgent,
    PlanExecuteAgent,
    ReflectionAgent,
    create_agent,
)
from reclab_agent.tools import EvidenceToolbox, Tool, ToolRegistry, ToolSpec, Toolbox

__all__ = [
    "ModelTurn", "OpenAIResponsesBackend", "EvidenceToolbox",
    "load_tasks", "run_reference_suite", "load_trace", "replay_trace",
    "Agent", "AgentConfig", "ModelBackend", "ReActAgent", "SimpleAgent",
    "PlanExecuteAgent", "ReflectionAgent", "create_agent",
    "Tool", "ToolRegistry", "ToolSpec", "Toolbox",
]

