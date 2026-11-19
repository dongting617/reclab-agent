"""Explicit-corpus, evidence-preserving retrieval without required dependencies.

BM25 works out of the box. Hybrid retrieval and reranking require actual
callables supplied by the caller; missing models never silently fall back to
lexical retrieval. This module does not download models or scan repositories.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence


EmbeddingFunction = Callable[[list[str]], Sequence[Sequence[float]]]
RerankingFunction = Callable[[str, list[str]], Sequence[float]]


class RetrievalError(ValueError):
    """Retrieval configuration, corpus, or model failure with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_BLOCKED_PARTS = frozenset({
    "test", "tests", "label", "labels", "answer", "answers", "answer_keys",
    "ground_truth", "ground-truth", "trace", "traces", "holdout", "hidden",
})
_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)
_HAN_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")


def tokenize(text: str) -> list[str]:
    """Preserve identifiers and use characters/bigrams for unsegmented Chinese."""
    tokens = []
    for match in _TOKEN_PATTERN.finditer(text):
        word = match.group(0).lower()
        if _HAN_PATTERN.fullmatch(word):
            tokens.extend(word)
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word)
    return tokens


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, limit: int = 30, rank_constant: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked IDs deterministically; repeated IDs in one list count once."""
    scores: dict[str, float] = {}
    if limit < 1 or rank_constant < 0:
        raise RetrievalError("INVALID_FUSION", "RRF needs limit >= 1 and rank_constant >= 0")
    for ranking in rankings:
        seen = set()
        rank = 0
        for identity in ranking:
            if identity in seen:
                continue
            seen.add(identity)
            rank += 1
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _reject_links(path: Path) -> None:
    # Reject Windows junctions/reparse points as well as ordinary symlinks.
    for component in (path, *path.parents):
        if component.is_symlink():
            raise RetrievalError("UNSAFE_PATH", f"Linked paths are not allowed: {component}")
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if getattr(info, "st_file_attributes", 0) & 0x400:
            raise RetrievalError("UNSAFE_PATH", f"Reparse points are not allowed: {component}")


def _vectors(values: Sequence[Sequence[float]], expected_count: int) -> list[list[float]]:
    try:
        rows = [[float(value) for value in row] for row in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise RetrievalError("INVALID_EMBEDDINGS", "Embeddings must be a numeric matrix") from error
    if len(rows) != expected_count:
        raise RetrievalError("INVALID_EMBEDDINGS", "Embedding row count does not match input count")
    dimension = len(rows[0]) if rows else 0
    for row in rows:
        if (not dimension or len(row) != dimension
                or not all(math.isfinite(value) for value in row)
                or not 0 < math.hypot(*row) < math.inf):
            raise RetrievalError("INVALID_EMBEDDINGS", "Embedding vectors must have equal nonzero dimensions and finite nonzero norms")
    return rows


class KnowledgeIndex:
    """An in-memory index over an explicit list of UTF-8 Markdown/text files.

    ``build(paths)`` is always explicit; directories and globs are not expanded.
    ``allowed_paths``, if supplied, further restricts the build manifest.
    ``embedding_fn`` receives a batch of strings and returns a dense matrix.
    ``reranker`` receives (query, document texts) and returns aligned scores.
    """

    def __init__(
        self,
        corpus_root: str | Path,
        allowed_paths: Iterable[str] | None = None,
        *,
        excluded_parts: Iterable[str] = (),
        chunk_chars: int = 500,
        overlap_chars: int = 80,
        embedding_fn: EmbeddingFunction | None = None,
        reranker: RerankingFunction | None = None,
    ):
        self.root = Path(os.path.abspath(os.fspath(corpus_root)))
        _reject_links(self.root)
        if not self.root.is_dir():
            raise RetrievalError("ROOT_NOT_FOUND", f"Corpus root does not exist: {self.root}")
        if chunk_chars < 100 or not 0 <= overlap_chars < chunk_chars // 2:
            raise RetrievalError("INVALID_CHUNKING", "Use chunk_chars >= 100 and overlap_chars < half the chunk size")
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.embedding_fn = embedding_fn
        self.reranker = reranker
        self.excluded_parts = _BLOCKED_PARTS | frozenset(str(part).lower() for part in excluded_parts)
        self._allowed = None
        if allowed_paths is not None:
            self._allowed = frozenset(self._source(path)[1] for path in allowed_paths)
        self.chunks: list[dict] = []
        self._counts: list[Counter] = []
        self._document_frequency: Counter = Counter()
        self._average_length = 0.0
        self._dense_vectors: list[list[float]] | None = None

    def _source(self, value: str | Path) -> tuple[Path, str]:
        if any(character in os.fspath(value) for character in "*?[]"):
            raise RetrievalError("INVALID_MANIFEST", "Source paths must be explicit, without wildcard patterns")
        candidate = Path(value)
        path = Path(os.path.abspath(os.fspath(candidate if candidate.is_absolute() else self.root / candidate)))
        try:
            relative = path.relative_to(self.root)
        except ValueError as error:
            raise RetrievalError("PATH_OUTSIDE_CORPUS", "Source must stay within corpus_root") from error
        _reject_links(path)
        name = relative.as_posix()
        for part in relative.parts:
            normalized = part.lower()
            stem = Path(normalized).stem
            pieces = set(re.split(r"[^a-z0-9]+", stem))
            if normalized in self.excluded_parts or stem in self.excluded_parts or pieces & self.excluded_parts:
                raise RetrievalError("EXCLUDED_SOURCE", f"Source contains a prohibited path component: {name}")
        if path.suffix.lower() not in {".md", ".txt"}:
            raise RetrievalError("UNSUPPORTED_SOURCE", "Only explicit .md and .txt files can be indexed")
        if self._allowed is not None and name not in self._allowed:
            raise RetrievalError("SOURCE_NOT_ALLOWED", f"Source is not in allowed_paths: {name}")
        if not path.is_file():
            raise RetrievalError("SOURCE_NOT_FOUND", f"Source is not a regular file: {name}")
        return path, name

    def _chunk(self, text: str, source: str, digest: str) -> list[dict]:
        lines = text.splitlines(keepends=True)
        starts = []
        offset = 0
        sections = []
        section_start = 0
        heading = ""
        fenced = False
        for line in lines:
            starts.append(offset)
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
            match = None if fenced else re.match(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", line.rstrip("\r\n"))
            if match:
                if offset > section_start:
                    sections.append((section_start, offset, heading))
                section_start, heading = offset, match.group(1)
            offset += len(line)
        if offset > section_start:
            sections.append((section_start, offset, heading))

        chunks = []
        for section_start, section_end, heading in sections:
            start = section_start
            while start < section_end:
                end = min(start + self.chunk_chars, section_end)
                if end < section_end:
                    # Prefer paragraph then line/word boundaries in the latter half.
                    lower = start + self.chunk_chars // 2
                    for delimiter in ("\n\n", "\n", " "):
                        boundary = text.rfind(delimiter, lower, end)
                        if boundary != -1:
                            end = boundary + len(delimiter)
                            break
                content_start, content_end = start, end
                while content_start < content_end and text[content_start].isspace():
                    content_start += 1
                while content_end > content_start and text[content_end - 1].isspace():
                    content_end -= 1
                if content_start < content_end:
                    content = text[content_start:content_end]
                    line_start = bisect_right(starts, content_start)
                    line_end = bisect_right(starts, content_end - 1)
                    chunk_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    identity = f"{source}\0{digest}\0{content_start}\0{content_end}\0{chunk_digest}"
                    chunks.append({
                        "chunk_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                        "source_relative": source,
                        "sha256": digest,
                        "chunk_sha256": chunk_digest,
                        "line_start": line_start,
                        "line_end": line_end,
                        "heading": heading,
                        "text": content,
                    })
                if end == section_end:
                    break
                start = max(start + 1, end - self.overlap_chars)
        return chunks

    def build(self, paths: list[str]) -> int:
        """Replace the index atomically from an explicit file manifest."""
        if isinstance(paths, (str, bytes)):
            raise RetrievalError("INVALID_MANIFEST", "build expects a list of explicit source paths")
        sources = {}
        for value in paths:
            path, name = self._source(value)
            sources[name] = path
        chunks = []
        for name, path in sorted(sources.items()):
            data = path.read_bytes()
            # Reject a linked path introduced while reading the manifest.
            _reject_links(path)
            try:
                text = data.decode("utf-8-sig").replace("\r\n", "\n")
            except UnicodeDecodeError as error:
                raise RetrievalError("INVALID_ENCODING", f"Source must be UTF-8: {name}") from error
            chunks.extend(self._chunk(text, name, hashlib.sha256(data).hexdigest()))
        counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        frequency = Counter()
        for count in counts:
            frequency.update(count.keys())
        self.chunks = chunks
        self._counts = counts
        self._document_frequency = frequency
        self._average_length = sum(sum(count.values()) for count in counts) / len(counts) if counts else 0.0
        self._dense_vectors = None
        return len(chunks)

    def _bm25(self, query: str) -> list[tuple[int, float]]:
        terms = Counter(tokenize(query))
        scored = []
        total = len(self.chunks)
        for index, count in enumerate(self._counts):
            length = sum(count.values())
            score = 0.0
            for term, query_count in terms.items():
                term_count = count.get(term, 0)
                if not term_count:
                    continue
                df = self._document_frequency[term]
                inverse_frequency = math.log1p((total - df + 0.5) / (df + 0.5))
                normalization = 1.5 * (1 - 0.75 + 0.75 * length / (self._average_length or 1))
                score += inverse_frequency * term_count * 2.5 / (term_count + normalization) * (1 + math.log(query_count))
            if score > 0:
                scored.append((index, score))
        return sorted(scored, key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]))

    def _dense(self, query: str) -> list[tuple[int, float]]:
        assert self.embedding_fn is not None
        try:
            if self._dense_vectors is None:
                self._dense_vectors = _vectors(self.embedding_fn([chunk["text"] for chunk in self.chunks]), len(self.chunks))
            query_vector = _vectors(self.embedding_fn([query]), 1)[0]
        except RetrievalError:
            raise
        except Exception as error:
            raise RetrievalError("EMBEDDING_FAILED", "Embedding callable failed") from error
        if len(query_vector) != len(self._dense_vectors[0]):
            raise RetrievalError("INVALID_EMBEDDINGS", "Query and document embedding dimensions differ")
        query_norm = math.hypot(*query_vector)
        scores = []
        for index, vector in enumerate(self._dense_vectors):
            # Normalize first to avoid overflow in the dot product.
            norm = math.hypot(*vector)
            score = math.fsum((left / query_norm) * (right / norm) for left, right in zip(query_vector, vector))
            scores.append((index, max(-1.0, min(1.0, score))))
        return sorted(scores, key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]))

    def search(self, query: str, k: int = 5, mode: str = "bm25") -> list[dict]:
        """Return chunks with citations and actual lexical/fused/reranked scores."""
        if mode not in {"bm25", "hybrid", "reranked"}:
            raise RetrievalError("INVALID_MODE", "mode must be bm25, hybrid, or reranked")
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise RetrievalError("INVALID_K", "k must be a positive integer")
        if mode in {"hybrid", "reranked"} and self.embedding_fn is None:
            raise RetrievalError("EMBEDDING_NOT_CONFIGURED", f"{mode} requires an embedding_fn")
        if mode == "reranked" and self.reranker is None:
            raise RetrievalError("RERANKER_NOT_CONFIGURED", "reranked mode requires a reranker")
        if not isinstance(query, str) or not query.strip() or not self.chunks:
            return []
        bm25 = self._bm25(query)
        if mode == "bm25":
            return [dict(self.chunks[index], mode=mode, score=score, bm25_score=score)
                    for index, score in bm25[:k]]

        dense = self._dense(query)
        lexical_scores = dict(bm25)
        dense_scores = dict(dense)
        identities = {chunk["chunk_id"]: index for index, chunk in enumerate(self.chunks)}
        fused = reciprocal_rank_fusion([
            [self.chunks[index]["chunk_id"] for index, _ in bm25[:20]],
            [self.chunks[index]["chunk_id"] for index, _ in dense[:20]],
        ], limit=30)
        results = [dict(
            self.chunks[identities[identity]], mode=mode, score=score,
            rrf_score=score, bm25_score=lexical_scores.get(identities[identity], 0.0),
            dense_score=dense_scores[identities[identity]],
        ) for identity, score in fused]
        if mode == "reranked":
            assert self.reranker is not None
            try:
                scores = [float(score) for score in self.reranker(query, [result["text"] for result in results])]
            except Exception as error:
                raise RetrievalError("RERANKER_FAILED", "Reranker callable failed") from error
            if len(scores) != len(results) or not all(math.isfinite(score) for score in scores):
                raise RetrievalError("INVALID_RERANKING", "Reranker must return one finite score per candidate")
            results = [dict(result, score=score, rerank_score=score) for result, score in zip(results, scores)]
            results.sort(key=lambda result: (-result["score"], -result["rrf_score"], result["chunk_id"]))
        return results[:k]
