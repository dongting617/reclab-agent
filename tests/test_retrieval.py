import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from reclab_agent.memory.retrieval import (
    KnowledgeIndex,
    RetrievalError,
    reciprocal_rank_fusion,
    tokenize,
)


class KnowledgeIndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="reclab-retrieval-")
        self.root = Path(self.directory.name)
        self.write("model.md", "# SASRec 模型\n\n低频商品可以使用采样权重。\n\n配置变量 sample_weight 和 max_seq_len。\n")
        self.write("training.txt", "训练过程中先验证数据划分，再比较学习率。\n训练负结果也应保留。\n")

    def tearDown(self):
        self.directory.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_chinese_and_english_identifier_retrieval(self):
        index = KnowledgeIndex(self.root)
        self.assertEqual(index.build(["model.md", "training.txt"]), 2)
        for query in ("低频商品", "SASRec", "sample_weight", "max_seq_len"):
            with self.subTest(query=query):
                results = index.search(query)
                self.assertEqual(results[0]["source_relative"], "model.md")
                self.assertGreater(results[0]["score"], 0)
                self.assertEqual(results[0]["mode"], "bm25")
        self.assertIn("sample_weight", tokenize("sample_weight"))
        self.assertIn("商品", tokenize("低频商品"))
        self.assertEqual(index.search("completely_unknown_identifier"), [])

    def test_only_explicit_manifest_is_indexed(self):
        self.write("decoy.md", "exclusive_decoy_text")
        self.write("tests/answers.md", "exclusive_answer")
        index = KnowledgeIndex(self.root)
        index.build(["model.md"])
        self.assertEqual(index.search("exclusive_decoy_text"), [])
        self.assertEqual({chunk["source_relative"] for chunk in index.chunks}, {"model.md"})

    def test_constructor_allowlist_adds_a_boundary(self):
        index = KnowledgeIndex(self.root, allowed_paths=["model.md"])
        index.build(["model.md"])
        with self.assertRaises(RetrievalError) as caught:
            index.build(["training.txt"])
        self.assertEqual(caught.exception.code, "SOURCE_NOT_ALLOWED")
        self.assertTrue(index.search("sample_weight"))

    def test_evidence_identity_lines_and_hash_are_stable(self):
        index = KnowledgeIndex(self.root)
        index.build(["training.txt", "model.md", "model.md"])
        first = index.search("sample_weight")[0]
        index.build(["model.md", "training.txt"])
        second = index.search("sample_weight")[0]
        self.assertEqual(first, second)
        self.assertEqual(first["sha256"], hashlib.sha256((self.root / "model.md").read_bytes()).hexdigest())
        self.assertEqual(first["chunk_sha256"], hashlib.sha256(first["text"].encode("utf-8")).hexdigest())
        self.assertEqual((first["line_start"], first["line_end"]), (1, 5))
        self.assertEqual(first["heading"], "SASRec 模型")
        self.write("model.md", "# SASRec 模型\n\nsample_weight 修改后的记录。\n")
        index.build(["model.md"])
        self.assertNotEqual(first["chunk_id"], index.search("sample_weight")[0]["chunk_id"])

    def test_heading_boundaries_and_chunk_limit(self):
        content = "# 第一部分\n\n" + ("推荐模型的训练过程和验证结果。" * 30)
        content += "\n\n# 第二部分\n\n" + ("another_variable and ranking metrics. " * 20)
        self.write("long.md", content)
        index = KnowledgeIndex(self.root)
        index.build(["long.md"])
        self.assertGreaterEqual(len(index.chunks), 3)
        for chunk in index.chunks:
            self.assertLessEqual(len(chunk["text"]), 500)
            self.assertLessEqual(chunk["line_start"], chunk["line_end"])
            self.assertFalse("第一部分" in chunk["text"] and "第二部分" in chunk["text"])
        self.assertEqual({chunk["heading"] for chunk in index.chunks}, {"第一部分", "第二部分"})

    def test_parent_escape_and_absolute_outside_paths_are_rejected(self):
        index = KnowledgeIndex(self.root)
        for path in ("../outside.md", str(self.root.parent / "outside.md")):
            with self.subTest(path=path), self.assertRaises(RetrievalError) as caught:
                index.build([path])
            self.assertEqual(caught.exception.code, "PATH_OUTSIDE_CORPUS")

    def test_symlink_file_directory_and_root_are_rejected(self):
        self.write("folder/info.md", "retrieval information")
        try:
            (self.root / "link.md").symlink_to(self.root / "model.md")
            (self.root / "linked-folder").symlink_to(self.root / "folder", target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"System cannot create symlinks: {error}")
        index = KnowledgeIndex(self.root)
        for name in ("link.md", "linked-folder/info.md"):
            with self.subTest(path=name), self.assertRaises(RetrievalError) as caught:
                index.build([name])
            self.assertEqual(caught.exception.code, "UNSAFE_PATH")
        with self.assertRaises(RetrievalError):
            KnowledgeIndex(self.root / "linked-folder")

    def test_sensitive_filenames_cannot_be_whitelisted(self):
        for name in ("labels.md", "test_answers.md", "traces/run.txt", "ground_truth.md"):
            self.write(name, "must not be indexed")
            with self.subTest(path=name), self.assertRaises(RetrievalError) as caught:
                KnowledgeIndex(self.root, allowed_paths=[name])
            self.assertEqual(caught.exception.code, "EXCLUDED_SOURCE")

    def test_symlink_and_reparse_guards_without_link_creation_privilege(self):
        index = KnowledgeIndex(self.root)
        source = self.root / "model.md"
        with patch.object(Path, "is_symlink", lambda path: path == source):
            with self.assertRaises(RetrievalError) as caught:
                index.build(["model.md"])
        self.assertEqual(caught.exception.code, "UNSAFE_PATH")
        actual_lstat = Path.lstat

        def reparse_lstat(path):
            if path == source:
                return SimpleNamespace(st_mode=0o100644, st_file_attributes=0x400)
            return actual_lstat(path)

        with patch.object(Path, "lstat", reparse_lstat):
            with self.assertRaises(RetrievalError) as caught:
                index.build(["model.md"])
        self.assertEqual(caught.exception.code, "UNSAFE_PATH")

    def test_custom_exclusion_adds_to_default_policy(self):
        self.write("private/note.md", "private note")
        index = KnowledgeIndex(self.root, excluded_parts=["private"])
        with self.assertRaises(RetrievalError) as caught:
            index.build(["private/note.md"])
        self.assertEqual(caught.exception.code, "EXCLUDED_SOURCE")

    def test_unsupported_and_missing_sources_fail(self):
        self.write("model.json", "{}")
        index = KnowledgeIndex(self.root)
        for name, code in (("model.json", "UNSUPPORTED_SOURCE"), ("absent.md", "SOURCE_NOT_FOUND"), ("*.md", "INVALID_MANIFEST")):
            with self.subTest(path=name), self.assertRaises(RetrievalError) as caught:
                index.build([name])
            self.assertEqual(caught.exception.code, code)
        with self.assertRaises(RetrievalError) as caught:
            index.build("model.md")
        self.assertEqual(caught.exception.code, "INVALID_MANIFEST")

    def test_models_are_required_for_the_named_mode(self):
        index = KnowledgeIndex(self.root)
        index.build(["model.md"])
        for mode in ("hybrid", "reranked"):
            with self.subTest(mode=mode), self.assertRaises(RetrievalError) as caught:
                index.search("SASRec", mode=mode)
            self.assertEqual(caught.exception.code, "EMBEDDING_NOT_CONFIGURED")
        index = KnowledgeIndex(self.root, embedding_fn=lambda texts: [[1.0] for _ in texts])
        index.build(["model.md"])
        with self.assertRaises(RetrievalError) as caught:
            index.search("SASRec", mode="reranked")
        self.assertEqual(caught.exception.code, "RERANKER_NOT_CONFIGURED")

    def test_dense_retrieval_recovers_lexically_unmatched_document(self):
        self.write("semantic.md", "An unrelated surface phrase with the relevant meaning.")
        self.write("lexical.md", "needle lexical match")
        calls = []

        def embed(texts):
            calls.append(list(texts))
            return [[1.0, 0.0] if text.startswith(("An unrelated", "needle query")) else [0.0, 1.0] for text in texts]

        index = KnowledgeIndex(self.root, embedding_fn=embed)
        index.build(["semantic.md", "lexical.md"])
        lexical = index.search("needle query")
        self.assertEqual([result["source_relative"] for result in lexical], ["lexical.md"])
        hybrid = index.search("needle query", mode="hybrid")
        relevant = next(result for result in hybrid if result["source_relative"] == "semantic.md")
        self.assertEqual(relevant["bm25_score"], 0)
        self.assertEqual(relevant["dense_score"], 1)
        self.assertEqual(relevant["mode"], "hybrid")
        index.search("needle query", mode="hybrid")
        self.assertEqual(len(calls), 3)  # One corpus batch, then two query batches.

    def test_reranker_uses_real_scores_and_can_change_order(self):
        index = KnowledgeIndex(
            self.root,
            embedding_fn=lambda texts: [[1.0, 0.0] for _ in texts],
            reranker=lambda query, texts: [10.0 if "负结果" in text else -1.0 for text in texts],
        )
        index.build(["model.md", "training.txt"])
        results = index.search("SASRec", mode="reranked")
        self.assertEqual(results[0]["source_relative"], "training.txt")
        self.assertEqual(results[0]["score"], 10.0)
        self.assertIn("rrf_score", results[0])
        self.assertEqual(results[0]["mode"], "reranked")

    def test_hybrid_pool_is_at_most_30(self):
        paths = []
        for index in range(45):
            name = f"doc-{index:02}.md"
            self.write(name, f"shared_keyword item_{index}")
            paths.append(name)
        index = KnowledgeIndex(self.root, embedding_fn=lambda texts: [[1.0] for _ in texts])
        index.build(paths)
        self.assertLessEqual(len(index.search("shared_keyword", k=100, mode="hybrid")), 30)

    def test_embedding_and_reranking_output_validation(self):
        for bad in ([], [[0.0, 0.0]], [[float("nan"), 1.0]], [["not_numeric"]]):
            index = KnowledgeIndex(self.root, embedding_fn=lambda texts, bad=bad: bad)
            index.build(["model.md"])
            with self.subTest(bad=bad), self.assertRaises(RetrievalError) as caught:
                index.search("SASRec", mode="hybrid")
            self.assertEqual(caught.exception.code, "INVALID_EMBEDDINGS")
        index = KnowledgeIndex(self.root, embedding_fn=lambda texts: [[1.0] for _ in texts], reranker=lambda query, texts: [])
        index.build(["model.md"])
        with self.assertRaises(RetrievalError) as caught:
            index.search("SASRec", mode="reranked")
        self.assertEqual(caught.exception.code, "INVALID_RERANKING")

    def test_invalid_mode_and_k(self):
        index = KnowledgeIndex(self.root)
        with self.assertRaises(RetrievalError) as caught:
            index.search("query", mode="imaginary")
        self.assertEqual(caught.exception.code, "INVALID_MODE")
        for k in (0, -1, True, "5"):
            with self.subTest(k=k), self.assertRaises(RetrievalError):
                index.search("query", k=k)

    def test_empty_manifest_and_empty_query(self):
        index = KnowledgeIndex(self.root)
        index.build(["model.md"])
        self.assertEqual(index.search("  "), [])
        self.assertEqual(index.build([]), 0)
        self.assertEqual(index.search("SASRec"), [])


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_deterministic_ties_and_duplicate_ids(self):
        first = reciprocal_rank_fusion([["b", "a", "a"], ["a", "b"]])
        second = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        self.assertEqual(first, second)
        self.assertEqual([identity for identity, _ in first], ["a", "b"])
        self.assertEqual(len(reciprocal_rank_fusion([[str(i) for i in range(100)]])), 30)


if __name__ == "__main__":
    unittest.main()
