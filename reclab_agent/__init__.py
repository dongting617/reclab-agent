"""RecLab Agent: a verifiable tool environment for recommendation experiments."""

from .benchmark import load_tasks, run_reference_suite
from .evaluation import load_trace, replay_trace
from .core import Agent, AgentConfig, ModelBackend, ModelTurn, OpenAIResponsesBackend
from .agents import ReActAgent, SimpleAgent, PlanExecuteAgent, ReflectionAgent, create_agent
from .tools import RecLabToolbox, Tool, ToolRegistry, ToolSpec, Toolbox

__all__ = [
    "ModelTurn", "OpenAIResponsesBackend", "RecLabToolbox",
    "load_tasks", "run_reference_suite", "load_trace", "replay_trace",
    "Agent", "AgentConfig", "ModelBackend", "ReActAgent", "SimpleAgent",
    "PlanExecuteAgent", "ReflectionAgent", "create_agent",
    "Tool", "ToolRegistry", "ToolSpec", "Toolbox",
]

