"""Knowledge function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.tools.group import ToolGroup

class KnowledgeTools(ToolGroup):
    definitions = (('search_knowledge',
      'Retrieve cited project experience and literature; read original scores for numeric '
      'claims.',
      {'query': {'type': 'string'}}),
     ('save_research_note',
      'Save a hypothesis or interpretation, explicitly not a verified measurement.',
      {'note_id': {'type': 'string'}, 'text': {'type': 'string'}}))

    def search_knowledge(self, query: str) -> dict[str, Any]:
        self.check_deadline()
        return {"matches": self.project.search(self.scope, query)}


    def save_research_note(self, note_id: str, text: str) -> dict[str, Any]:
        self.check_deadline()
        return self.project.note("experience", self.scope, note_id,
                                 "# Agent research note (unverified interpretation)\n\n" + text)
