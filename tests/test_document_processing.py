"""
tests/test_document_processing.py
──────────────────────────────────
Unit tests for document processing utilities.
"""

import json
import pathlib
import tempfile
import unittest

from src.document_processing.txt_processor import extract_txt
from src.document_processing.text_chunker import chunk_text, chunk_documents


class TestTxtProcessor(unittest.TestCase):
    def test_basic_extraction(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w",
                                        encoding="utf-8", delete=False) as f:
            f.write("Hello world.\n\nThis is a test document.")
            path = f.name
        result = extract_txt(path)
        pathlib.Path(path).unlink()
        self.assertFalse(result.error)
        self.assertIn("Hello world", result.text)
        self.assertGreater(result.word_count, 0)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w",
                                        encoding="utf-8", delete=False) as f:
            path = f.name
        result = extract_txt(path)
        pathlib.Path(path).unlink()
        self.assertTrue(result.error)

    def test_unicode(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w",
                                        encoding="utf-8", delete=False) as f:
            f.write("Ärger mit Ümlauten 日本語 テスト")
            path = f.name
        result = extract_txt(path)
        pathlib.Path(path).unlink()
        self.assertIn("Ärger", result.text)


class TestTextChunker(unittest.TestCase):
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Short text.", "test.txt", max_tokens=1000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "test.txt")

    def test_long_text_multiple_chunks(self):
        # Generate ~5000 chars
        long_text = ("This is a sentence about electronics. " * 200)
        chunks = chunk_text(long_text, "test.txt", max_tokens=100, overlap_chars=20)
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertEqual(ch.source, "test.txt")
            self.assertIsInstance(ch.chunk_index, int)

    def test_chunk_documents(self):
        docs = [
            {"filename": "a.txt", "text": "Doc A content. " * 50},
            {"filename": "b.txt", "text": "Doc B content. " * 50},
        ]
        chunks = chunk_documents(docs, max_tokens=50)
        sources = {ch.source for ch in chunks}
        self.assertIn("a.txt", sources)
        self.assertIn("b.txt", sources)


if __name__ == "__main__":
    unittest.main()
