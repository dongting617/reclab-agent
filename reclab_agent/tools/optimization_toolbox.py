"""Optimization toolbox assembly; business handlers live in capability groups."""
from __future__ import annotations
from reclab_agent.tools.study_context import StudyContext
from reclab_agent.tools.registry import ToolRegistry
from reclab_agent.tools.workspace import WorkspaceTools
from reclab_agent.tools.planning import PlanningTools
from reclab_agent.tools.experiments import ExperimentTools
from reclab_agent.tools.diagnostics import DiagnosticTools
from reclab_agent.tools.baselines import BaselineTools
from reclab_agent.tools.evaluation import EvaluationTools
from reclab_agent.tools.knowledge import KnowledgeTools
from reclab_agent.tools.literature import LiteratureTools

class OptimizationToolbox:
    def __init__(self, *args, **kwargs):
        self.context = StudyContext(*args, **kwargs)
        self._tools = ToolRegistry()
        self.groups = {
            "workspace": WorkspaceTools(self.context),
            "planning": PlanningTools(self.context),
            "experiments": ExperimentTools(self.context),
            "diagnostics": DiagnosticTools(self.context),
            "baselines": BaselineTools(self.context),
            "evaluation": EvaluationTools(self.context),
            "knowledge": KnowledgeTools(self.context),
            "literature": LiteratureTools(self.context)
        }
        for name in ("workspace", "planning", "diagnostics", "baselines", "evaluation"):
            self.groups[name].register(self._tools)
        names = {"read_log"}
        if self.context._runner is not None:
            names.update({"run_experiment", "get_experiment"})
        self.groups["experiments"].register(self._tools, names)

    @property
    def specs(self):
        return self._tools.specs

    def invoke(self, name, arguments=None):
        return self._tools.invoke(name, arguments)

    def attach_knowledge(self, literature):
        self.context.literature = literature
        for name in ("knowledge", "literature"):
            self.groups[name].register(self._tools)
        self.groups["experiments"].register(self._tools, {"list_experiments"})

    def __getattr__(self, name):
        context = object.__getattribute__(self, "context")
        if name in context.operations:
            return context.operations[name]
        return getattr(context, name)

    def __setattr__(self, name, value):
        if name not in {"context", "_tools", "groups"} and "context" in self.__dict__ and hasattr(self.context, name):
            setattr(self.context, name, value)
        else:
            object.__setattr__(self, name, value)
