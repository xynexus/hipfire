import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("http_embeddings.py")
SPEC = importlib.util.spec_from_file_location("http_embeddings", MODULE_PATH)
http_embeddings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(http_embeddings)


class HttpEmbeddingsBenchmarkTests(unittest.TestCase):
    def test_truncate_utf8_respects_byte_limit_and_character_boundaries(self):
        text = "alpha beta gamma δelta"
        truncated = http_embeddings.truncate_utf8(text, 14)
        self.assertEqual(truncated, "alpha beta")
        self.assertLessEqual(len(truncated.encode("utf-8")), 14)

    def test_load_documents_combines_title_and_text_and_skips_empty_rows(self):
        rows = [
            {"title": "A title", "text": "Document body"},
            {"title": "", "text": "  "},
            {"text": "Text only"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            documents = http_embeddings.load_documents(corpus, max_bytes=1_024)

        self.assertEqual(documents, ["A title\nDocument body", "Text only"])

    def test_percentile_uses_nearest_rank_interpolation(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(http_embeddings.percentile(values, 0.0), 1.0)
        self.assertEqual(http_embeddings.percentile(values, 0.5), 2.5)
        self.assertEqual(http_embeddings.percentile(values, 1.0), 4.0)


if __name__ == "__main__":
    unittest.main()
