"""Planning function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.tools.contracts import ToolError
from reclab_agent.tools.tool_group import ToolGroup

class PlanningTools(ToolGroup):
    definitions = (('record_hypothesis',
      'Register a testable hypothesis before candidate editing and training.',
      {'statement': {'type': 'string'},
       'changes': {'type': 'string'},
       'expected': {'type': 'string'},
       'risks': {'type': 'string'},
       'evidence': {'type': 'array', 'items': {'type': 'string'}},
       'process_metrics': {'type': 'array',
                           'items': {'type': 'object',
                                     'properties': {'name': {'type': 'string'},
                                                    'purpose': {'type': 'string',
                                                                'enum': ['diagnostic',
                                                                         'required']},
                                                    'expected': {'type': 'string'},
                                                    'operator': {'type': ['string', 'null'],
                                                                 'enum': ['>',
                                                                          '>=',
                                                                          '<',
                                                                          '<=',
                                                                          '==',
                                                                          '!=',
                                                                          None]},
                                                    'threshold': {'type': ['number', 'null']}},
                                     'required': ['name',
                                                  'purpose',
                                                  'expected',
                                                  'operator',
                                                  'threshold'],
                                     'additionalProperties': False}}}),)

    def record_hypothesis(self, statement: str, changes: str, expected: str, risks: str,
                          evidence: list[str], process_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if self.active_stage not in {None, "hypothesis"}:
            raise ToolError("WRONG_STAGE", "hypotheses can only be registered in the hypothesis stage")
        for value in (statement, changes, expected, risks):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("hypothesis requires statement, changes, expected and risks")
        if not isinstance(evidence, list) or any(not isinstance(x, str) for x in evidence):
            raise ValueError("evidence must contain reference strings")
        if not isinstance(process_metrics, list) or any(not isinstance(x, dict) for x in process_metrics):
            raise ValueError("process_metrics must contain metric definitions")
        from reclab_agent.experiments.diagnostics import validate_definitions
        validate_definitions(process_metrics)
        self.hypothesis = {"statement": statement, "changes": changes, "expected": expected,
                           "risks": risks, "evidence": evidence, "process_metrics": process_metrics}
        self.save_state()
        return self.hypothesis
