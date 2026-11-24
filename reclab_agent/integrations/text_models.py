"""Optional CPU-only text models behind existing embedding/reranker callables."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from importlib.metadata import PackageNotFoundError, version


@dataclass
class TextModels:
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_revision: str | None = None
    reranking_revision: str | None = None
    local_files_only: bool = True

    @cached_property
    def encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ValueError('install CPU text-model dependencies with pip install -e ".[retrieval]"') from error
        return SentenceTransformer(self.embedding_model, device="cpu", trust_remote_code=False,
                                   revision=self.embedding_revision, local_files_only=self.local_files_only)

    @cached_property
    def ranker(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise ValueError('install CPU text-model dependencies with pip install -e ".[retrieval]"') from error
        return CrossEncoder(self.reranking_model, device="cpu", trust_remote_code=False,
                            revision=self.reranking_revision, local_files_only=self.local_files_only)

    def embed(self, texts):
        return self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def rerank(self, query, texts):
        if not texts:
            return []
        return self.ranker.predict([(query, text) for text in texts], show_progress_bar=False).tolist()

    def metadata(self):
        def installed(name):
            try:
                return version(name)
            except PackageNotFoundError:
                return None
        def revision(model):
            # Hugging Face records the resolved commit on the loaded config.
            base = getattr(model, "model", None)
            if base is None:
                try:
                    base = model[0].auto_model
                except (TypeError, AttributeError, IndexError):
                    return None
            return getattr(getattr(base, "config", None), "_commit_hash", None)
        return {"device": "cpu", "embedding_model": self.embedding_model,
                "libraries": {name: installed(name) for name in ("torch", "sentence-transformers")},
                "embedding_loaded": "encoder" in self.__dict__, "reranking_loaded": "ranker" in self.__dict__,
                "embedding_revision": self.embedding_revision,
                "embedding_resolved_revision": revision(self.encoder) if "encoder" in self.__dict__ else None,
                "reranking_model": self.reranking_model,
                "reranking_revision": self.reranking_revision,
                "reranking_resolved_revision": revision(self.ranker) if "ranker" in self.__dict__ else None,
                "local_files_only": self.local_files_only}
