"""
src/dataset_generation/validators.py
──────────────────────────────────────
Parse, validate, and deduplicate generated QA records.
Never trust the LLM to return perfectly structured JSON.
"""

import json
import re
import hashlib
from dataclasses import dataclass, field


@dataclass
class ParseResult:
    accepted: list[dict] = field(default_factory=list)
    duplicates: int = 0
    rejected: int = 0
    parse_errors: list[str] = field(default_factory=list)


def _normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _fingerprint(question: str) -> str:
    return hashlib.md5(_normalize_str(question).encode()).hexdigest()


def parse_llm_response(raw: str) -> list[dict] | None:
    """
    Try to extract JSON examples from the raw LLM response.
    Attempts:
      1. Direct JSON parse of the whole response.
      2. Extract first JSON object via regex.
      3. Extract array via regex.
    Returns list of raw dicts or None on total failure.
    """
    text = raw.strip()

    # Attempt 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "examples" in obj:
            return obj["examples"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences and retry
    stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    stripped = re.sub(r"```\s*$", "", stripped, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "examples" in obj:
            return obj["examples"]
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass

    # Attempt 3: regex extract JSON object
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "examples" in obj:
                return obj["examples"]
        except json.JSONDecodeError:
            pass

    # Attempt 4: regex extract JSON array
    match = re.search(r"\[.*\]", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_and_deduplicate(
    raw_examples: list[dict],
    seen_fingerprints: set[str],
) -> ParseResult:
    """
    Validate each raw record and deduplicate by question fingerprint.
    Mutates seen_fingerprints in place.
    """
    result = ParseResult()

    for idx, item in enumerate(raw_examples):
        if not isinstance(item, dict):
            result.rejected += 1
            result.parse_errors.append(f"Item {idx}: not a dict — {type(item).__name__}")
            continue

        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not question:
            result.rejected += 1
            result.parse_errors.append(f"Item {idx}: empty question")
            continue
        if not answer:
            result.rejected += 1
            result.parse_errors.append(f"Item {idx}: empty answer (q={question[:60]!r})")
            continue
        if len(question) < 5:
            result.rejected += 1
            result.parse_errors.append(f"Item {idx}: question too short: {question!r}")
            continue
        if len(answer) < 5:
            result.rejected += 1
            result.parse_errors.append(f"Item {idx}: answer too short for: {question[:60]!r}")
            continue

        fp = _fingerprint(question)
        if fp in seen_fingerprints:
            result.duplicates += 1
            continue

        seen_fingerprints.add(fp)
        result.accepted.append({"question": question, "answer": answer})

    return result
