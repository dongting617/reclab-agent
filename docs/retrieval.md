# Explicit-corpus retrieval

`reclab_agent.retrieval.KnowledgeIndex` is a dependency-free BM25 foundation for
the RecLab research knowledge base. It indexes only the files passed to
`build`; it never walks a repository or discovers documents automatically.

```python
from reclab_agent.retrieval import KnowledgeIndex

index = KnowledgeIndex(
    "knowledge",
    allowed_paths=["models/sasrec.md", "protocol.md"],
)
index.build(["models/sasrec.md", "protocol.md"])
matches = index.search("低频商品 sample_weight", k=5, mode="bm25")
```

Files must be UTF-8 `.md` or `.txt` regular files under `corpus_root`. Explicit
allowlisting is required by the build manifest; `allowed_paths` can further
restrict the caller. Parent escapes, symlinks, Windows reparse points, and
paths containing test/answer/label/trace/holdout components are rejected. Extra
prohibited path components can be added with `excluded_parts`. This is a path
policy, not a content classifier: the curator must ensure allowed prose does
not contain hidden answers or test labels. Do not modify corpus files
concurrently while an index is being built.

Markdown headings define section boundaries; paragraph/line/word boundaries
are preferred when splitting long sections. Defaults are at most 500
characters per chunk with approximately 80 characters of overlap within a
section. Chinese characters and bigrams support unsegmented Chinese searches;
ASCII identifiers such as `sample_weight` remain intact.

Each result includes:

- `source_relative`: source path relative to the corpus root;
- `sha256`: SHA-256 of the original source bytes;
- `chunk_id` and `chunk_sha256`: stable chunk identity and content digest;
- `line_start`, `line_end`, `heading`, and `text`: one-based source citation;
- `mode`, `score`, and the scores used by the selected retrieval mode.

Rebuilding from the same bytes and configuration produces stable citations and
IDs. A changed source hash invalidates all of that source's previous chunk IDs.
The index is replaced only after every listed source has been validated/read.

## Dense retrieval and reranking

Hybrid mode takes the lexical top 20 and dense cosine top 20, fuses with RRF
(`1 / (60 + rank)`), and keeps at most 30 candidates. Reranked mode applies an
injected scorer to those candidates and returns the top `k` (default 5).

```python
def embedding_fn(texts: list[str]) -> list[list[float]]:
    # Run a configured local embedding model or a batched embedding endpoint.
    return configured_embedding_model(texts)

def reranker(query: str, texts: list[str]) -> list[float]:
    # Return one finite relevance score aligned with each input document.
    return configured_reranking_model(query, texts)

index = KnowledgeIndex(
    "knowledge",
    embedding_fn=embedding_fn,
    reranker=reranker,
)
index.build(["protocol.md"])
results = index.search("数据划分协议", mode="reranked")
```

Both document and query embeddings use the same batched `embedding_fn`.
Document vectors are cached until the next `build`; configure the callable's
model/version for the lifetime of the index. Callables must return one finite,
nonzero vector per input with a shared dimension. The implementation computes
cosine similarity in standard Python and does not require NumPy or FAISS.

`hybrid` without embeddings raises `EMBEDDING_NOT_CONFIGURED`; `reranked`
requires both embeddings and a reranker. Model execution/output errors are
explicit failures. There is no silent lexical fallback and no automatic model
download or package installation.

The committed tests exercise actual BM25 and the dense/fusion/reranking
algorithms using deterministic injected vectors/scores. No pretrained BGE
embedding/reranker, retrieval-quality benchmark, or remote embedding provider
has been run or validated by this implementation. Those are separate
integration and evaluation steps, not claimed capabilities of these tests.
