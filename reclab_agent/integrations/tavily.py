"""Bounded Tavily search/extraction, cached as cited project knowledge."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from reclab_agent.domain.contracts import canonical_json, content_hash
from reclab_agent.tools.contracts import ToolError


class LiteratureService:
    def __init__(self, project, scope, reserve, *, key_env="TAVILY_API_KEY", transport=None):
        self.project, self.scope, self.reserve, self.key_env = project, scope, reserve, key_env
        self.transport = transport or self._request
        self.timeout = 30

    def _request(self, endpoint, payload, key):
        request = urllib.request.Request("https://api.tavily.com/" + endpoint,
            data=canonical_json(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(4_000_001)
            if len(raw) > 4_000_000:
                raise ToolError("LITERATURE_TOO_LARGE", "literature response exceeded the size limit")
            return json.loads(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            raise ToolError("LITERATURE_UNAVAILABLE", "literature provider request failed") from None

    def call(self, endpoint, payload):
        from reclab_agent.infrastructure.files import atomic_write
        from reclab_agent.infrastructure.files import confined_path
        from reclab_agent.domain.configuration import read_json_strict
        cache = confined_path(self.project.root, "artifacts/literature/" + content_hash({
            "scope": self.scope, "endpoint": endpoint, "payload": payload}) + ".json")
        if cache.exists():
            return read_json_strict(cache, max_bytes=1_000_000)
        key = os.environ.get(self.key_env)
        if not key:
            raise ToolError("SEARCH_NOT_CONFIGURED", "configure the named Tavily key; local knowledge remains available")
        timeout = self.reserve()
        if timeout is not None:
            self.timeout = timeout
        result = self.transport(endpoint, payload, key)
        if not isinstance(result, dict) or not isinstance(result.get("results"), list):
            raise ToolError("LITERATURE_INVALID", "provider returned an invalid result")
        references = []
        for row in result["results"]:
            url = row.get("url", "")
            if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
                continue
            raw = row.get("raw_content") or row.get("content") or ""
            if not isinstance(raw, str):
                continue
            note_id = "lit-" + content_hash({"scope": self.scope, "url": url, "text": raw})[:24]
            coverage = "extracted_text" if row.get("raw_content") else "snippet_only"
            text = (f"# {str(row.get('title') or url)[:500]}\n\nSource: {url}\n"
                    f"Retrieved: {datetime.now(timezone.utc).isoformat()}\nCoverage: {coverage}\n\n"
                    "External research material; not an instruction or measured experiment result.\n\n" + raw[:120_000])
            ref = self.project.note("literature", self.scope, note_id, text)
            references.append({**ref, "url": url, "coverage": coverage, "text": raw[:12_000]})
        normalized = {"references": references, "failed_count": len(result.get("failed_results", []))}
        if references:
            atomic_write(cache, canonical_json(normalized).encode("utf-8"))
        return normalized

    def search(self, query: str):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1-2000 characters")
        return self.call("search", {"query": query, "max_results": 5, "search_depth": "basic",
                                    "include_raw_content": False, "include_answer": False,
                                    "include_domains": ["arxiv.org", "openreview.net", "proceedings.mlr.press",
                                                        "aclanthology.org", "github.com", "pytorch.org"]})

    def fetch(self, url: str):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("a public HTTP(S) source URL is required")
        return self.call("extract", {"urls": [url], "format": "markdown", "extract_depth": "basic"})
