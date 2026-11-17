"""
tests/test_dataset_generation.py
──────────────────────────────────
Unit tests for dataset validators and adapters.
"""

import unittest
from src.dataset_generation.validators import parse_llm_response, validate_and_deduplicate
from src.dataset_generation.dataset_adapters import (
    qa_to_messages, examples_to_jsonl, validate_jsonl_schema
)


class TestValidators(unittest.TestCase):
    def test_parse_clean_json(self):
        raw = '{"examples": [{"question": "What is X?", "answer": "X is Y."}]}'
        result = parse_llm_response(raw)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question"], "What is X?")

    def test_parse_with_code_fence(self):
        raw = '```json\n{"examples":[{"question":"Q1?","answer":"A1."}]}\n```'
        result = parse_llm_response(raw)
        self.assertIsNotNone(result)

    def test_parse_invalid_returns_none(self):
        raw = "This is not JSON at all."
        result = parse_llm_response(raw)
        self.assertIsNone(result)

    def test_deduplication(self):
        examples = [
            {"question": "What is X?", "answer": "X is the first item."},
            {"question": "What is X?", "answer": "X is described again."},  # same question -> duplicate
            {"question": "What is Y?", "answer": "Y is the second item."},
        ]
        seen: set = set()
        result = validate_and_deduplicate(examples, seen)
        self.assertEqual(len(result.accepted), 2, f"accepted={result.accepted}, errors={result.parse_errors}")
        self.assertEqual(result.duplicates, 1)

    def test_empty_question_rejected(self):
        examples = [{"question": "", "answer": "The answer is here."}]
        seen = set()
        result = validate_and_deduplicate(examples, seen)
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(result.rejected, 1)

    def test_empty_answer_rejected(self):
        examples = [{"question": "What is it?", "answer": ""}]
        seen = set()
        result = validate_and_deduplicate(examples, seen)
        self.assertEqual(len(result.accepted), 0)
        self.assertEqual(result.rejected, 1)


class TestDatasetAdapters(unittest.TestCase):
    def test_qa_to_messages_schema(self):
        record = qa_to_messages("Q?", "A.", "System.")
        self.assertIn("messages", record)
        roles = [m["role"] for m in record["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_examples_to_jsonl(self):
        examples = [{"question": "Q1?", "answer": "A1."}]
        records = examples_to_jsonl(examples)
        self.assertEqual(len(records), 1)
        msgs = records[0]["messages"]
        self.assertEqual(msgs[1]["content"], "Q1?")
        self.assertEqual(msgs[2]["content"], "A1.")

    def test_validate_jsonl_schema_valid(self):
        records = [{"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]}]
        errors = validate_jsonl_schema(records)
        self.assertEqual(errors, [])

    def test_validate_jsonl_schema_missing_messages(self):
        records = [{"instruction": "Q", "output": "A"}]
        errors = validate_jsonl_schema(records)
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
