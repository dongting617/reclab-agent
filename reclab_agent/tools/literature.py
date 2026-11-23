"""Literature function tools: schemas and handlers."""
from __future__ import annotations
from typing import Any
from reclab_agent.tools.group import ToolGroup

class LiteratureTools(ToolGroup):
    definitions = (('search_literature',
      'Search external research sources with Tavily; snippets do not establish experiment '
      'facts.',
      {'query': {'type': 'string'}}),
     ('fetch_literature',
      'Extract selected source text and save provenance to the project wiki.',
      {'url': {'type': 'string'}}))

    def search_literature(self, query: str) -> dict[str, Any]:
        return self.literature.search(query)

    def fetch_literature(self, url: str) -> dict[str, Any]:
        return self.literature.fetch(url)
