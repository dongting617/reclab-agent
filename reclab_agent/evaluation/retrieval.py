"""Versioned paragraph retrieval evaluation; answers stay outside the corpus."""
from __future__ import annotations

import hashlib
import time
from reclab_agent.domain.configuration import read_json_strict
from reclab_agent.domain.contracts import content_hash
from reclab_agent.infrastructure.files import confined_path, read_regular_file
from reclab_agent.memory.retrieval import KnowledgeIndex


def evaluate_retrieval(manifest_path, *, modes=("bm25",), models=None, k=5):
    from pathlib import Path
    path = Path(manifest_path).absolute()
    manifest = read_json_strict(path, max_bytes=2_000_000)
    if manifest.get("schema_version") != 1 or type(k) is not int or not 1 <= k <= 20:
        raise ValueError("retrieval evaluation needs schema_version=1 and k between 1 and 20")
    root = confined_path(path.parent, manifest["corpus_root"])
    documents = manifest["documents"]
    if not documents or len({d["path"] for d in documents}) != len(documents):
        raise ValueError("documents must be nonempty and unique")
    line_counts = {}
    for document in documents:
        raw = read_regular_file(confined_path(root, document["path"]), max_bytes=1_000_000)
        if hashlib.sha256(raw).hexdigest() != document["sha256"]:
            raise ValueError("retrieval corpus changed after annotation")
        line_counts[document["path"]] = len(raw.decode("utf-8").splitlines())
    index = KnowledgeIndex(root, allowed_paths=[d["path"] for d in documents],
                           chunk_chars=300, overlap_chars=40,
                           embedding_fn=models.embed if models else None,
                           reranker=models.rerank if models else None)
    index.build([d["path"] for d in documents])
    queries = manifest["queries"]
    if not queries or len({q["id"] for q in queries}) != len(queries):
        raise ValueError("queries must be nonempty and unique")
    known = {d["path"] for d in documents}
    for query in queries:
        for ref in query["relevant"]:
            if ref["source"] not in known or type(ref["line_start"]) is not int or type(ref["line_end"]) is not int or not 1 <= ref["line_start"] <= ref["line_end"] <= line_counts[ref["source"]]:
                raise ValueError("relevance labels must point to valid source paragraphs")
    runs = []
    for mode in modes:
        rows = []
        for query in queries:
            started = time.perf_counter()
            hits = index.search(query["query"], k=k, mode=mode)
            ranks = []
            for ref in query["relevant"]:
                rank = next((i for i, hit in enumerate(hits, 1)
                             if hit["source_relative"] == ref["source"]
                             and hit["line_start"] <= ref["line_end"]
                             and hit["line_end"] >= ref["line_start"]), None)
                ranks.append(rank)
            found = [rank for rank in ranks if rank is not None]
            rows.append({"id": query["id"], "answerable": bool(ranks),
                         "recall": len(found) / len(ranks) if ranks else None,
                         "reciprocal_rank": 1 / min(found) if found else 0,
                         "returned_any": bool(hits), "elapsed_ms": round((time.perf_counter()-started)*1000, 3),
                         "hits": [{key: h[key] for key in ("source_relative", "line_start", "line_end",
                                   "sha256", "chunk_id", "score")} for h in hits]})
        answerable = [r for r in rows if r["answerable"]]
        absent = [r for r in rows if not r["answerable"]]
        runs.append({"mode": mode, "queries": len(rows), "answerable_queries": len(answerable),
                     f"recall@{k}": sum(r["recall"] for r in answerable)/len(answerable) if answerable else None,
                     "mrr": sum(r["reciprocal_rank"] for r in answerable)/len(answerable) if answerable else None,
                     "no_answer_empty_rate": sum(not r["returned_any"] for r in absent)/len(absent) if absent else None,
                     "elapsed_ms": sum(r["elapsed_ms"] for r in rows), "results": rows})
    return {"kind": "paragraph_retrieval", "manifest_hash": content_hash(manifest),
            "models": models.metadata() if models else None, "runs": runs,
            "limitations": ["Retrieval metrics do not measure answer factual support.",
                           "Dense retrieval has no calibrated abstention threshold.",
                           "First-query latency includes lazy model loading."]}
