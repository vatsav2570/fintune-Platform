"""
src/dataset_generation/dataset_adapters.py
───────────────────────────────────────────
Convert generated QA pairs into the EXACT schema used by the existing trainers.

Canonical schema (from train.jsonl / train_9800.jsonl / train_9800_300.jsonl):

{"messages":[
  {"role":"system","content":"<system_prompt>"},
  {"role":"user","content":"<question>"},
  {"role":"assistant","content":"<answer>"}
]}

This is the ONLY schema emitted. Never use instruction/input/output format.
"""

import json
import pathlib
from typing import Callable


DEFAULT_SYSTEM_PROMPT = (
    "You are a knowledgeable and precise assistant. "
    "Answer questions accurately based on the provided training material. "
    "Be concise and factual."
)


def qa_to_messages(
    question: str,
    answer: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """Convert a single QA pair to the messages schema."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def examples_to_jsonl(
    examples: list[dict],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Convert a list of {question, answer} dicts to messages-format records.
    Returns the list of JSONL-ready dicts.
    """
    records = []
    for i, ex in enumerate(examples):
        record = qa_to_messages(ex["question"], ex["answer"], system_prompt)
        records.append(record)
        if progress_callback:
            progress_callback(i + 1, len(examples))
    return records


def save_jsonl(records: list[dict], path: str | pathlib.Path) -> None:
    """Write records to a JSONL file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: str | pathlib.Path) -> list[dict]:
    """Read and parse a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc
    return records


def validate_jsonl_schema(records: list[dict]) -> list[str]:
    """
    Validate that every record matches the messages schema.
    Returns a list of validation error strings (empty = valid).
    """
    errors = []
    for i, rec in enumerate(records):
        if "messages" not in rec:
            errors.append(f"Record {i}: missing 'messages' key")
            continue
        msgs = rec["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            errors.append(f"Record {i}: 'messages' must be a list with ≥2 items")
            continue
        for j, msg in enumerate(msgs):
            if "role" not in msg:
                errors.append(f"Record {i}, msg {j}: missing 'role'")
            if "content" not in msg:
                errors.append(f"Record {i}, msg {j}: missing 'content'")
    return errors
