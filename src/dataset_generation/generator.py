"""
src/dataset_generation/generator.py
────────────────────────────────────
Orchestrate batched dataset generation from document chunks.
Calls the Azure client, validates responses, tracks progress.
"""

import json
import logging
import math
import pathlib
import datetime
from typing import Callable

from src.dataset_generation.azure_client import chat_completion
from src.dataset_generation.generation_prompts import build_generation_prompt
from src.dataset_generation.validators import parse_llm_response, validate_and_deduplicate
from src.dataset_generation.dataset_adapters import (
    examples_to_jsonl, save_jsonl, DEFAULT_SYSTEM_PROMPT
)
from src.document_processing.text_chunker import Chunk
from src.config import GENERATION_BATCH_SIZE

logger = logging.getLogger(__name__)


def generate_dataset(
    chunks: list[Chunk],
    total_questions: int,
    question_types: list[str],
    difficulty: str,
    output_jsonl_path: str | pathlib.Path,
    output_metadata_path: str | pathlib.Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    custom_instructions: str = "",
    batch_size: int = GENERATION_BATCH_SIZE,
    progress_callback: Callable[[int, int, str], None] | None = None,
    existing_examples: list[dict] | None = None,
) -> dict:
    """
    Generate `total_questions` examples from `chunks` using Azure OpenAI.

    progress_callback(accepted_so_far, total_questions, status_message)

    Returns a summary dict with counts and file paths.
    """

    accepted: list[dict] = list(existing_examples or [])
    seen_fps: set[str] = set()
    total_duplicates = 0
    total_rejected = 0
    total_parse_errors: list[str] = []

    # Pre-seed seen fingerprints from existing examples
    from src.dataset_generation.validators import _fingerprint
    for ex in accepted:
        msgs = ex.get("messages", [])
        for m in msgs:
            if m.get("role") == "user":
                seen_fps.add(_fingerprint(m["content"]))

    remaining = total_questions - len(accepted)

    if not chunks:
        return {
            "error": "No document content to generate from.",
            "accepted": len(accepted),
            "duplicates": total_duplicates,
            "rejected": total_rejected,
        }

    if remaining <= 0:
        return {
            "accepted": len(accepted),
            "duplicates": 0,
            "rejected": 0,
            "already_complete": True,
        }

    # Distribute questions across chunks proportionally
    questions_per_chunk = max(1, math.ceil(remaining / len(chunks)))
    chunk_queue: list[tuple[Chunk, int]] = []
    allocated = 0
    for i, chunk in enumerate(chunks):
        if allocated >= remaining:
            break
        q_for_this = min(questions_per_chunk, remaining - allocated)
        chunk_queue.append((chunk, q_for_this))
        allocated += q_for_this

    # Generate in batches
    for chunk, n_questions_needed in chunk_queue:
        batches = math.ceil(n_questions_needed / batch_size)
        for b in range(batches):
            still_needed = total_questions - len(accepted)
            if still_needed <= 0:
                break
            this_batch = min(batch_size, still_needed)
            status = (
                f"Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}, "
                f"batch {b + 1}/{batches} — {len(accepted)}/{total_questions} accepted"
            )
            if progress_callback:
                progress_callback(len(accepted), total_questions, status)

            try:
                messages = build_generation_prompt(
                    chunk_text=chunk.text,
                    source_filename=chunk.source,
                    batch_size=this_batch,
                    question_types=question_types,
                    difficulty=difficulty,
                    custom_instructions=custom_instructions,
                )
                raw = chat_completion(messages, max_tokens=4096, temperature=0.8)
            except RuntimeError as exc:
                logger.error("Generation error: %s", exc)
                total_parse_errors.append(f"Batch error: {exc}")
                continue

            raw_examples = parse_llm_response(raw)
            if raw_examples is None:
                total_parse_errors.append(
                    f"Could not parse JSON from response (chunk {chunk.chunk_index})"
                )
                logger.warning("Failed to parse LLM response for chunk %d", chunk.chunk_index)
                continue

            result = validate_and_deduplicate(raw_examples, seen_fps)
            total_duplicates += result.duplicates
            total_rejected += result.rejected
            total_parse_errors.extend(result.parse_errors)

            # Convert accepted QA pairs to messages format
            new_records = examples_to_jsonl(result.accepted, system_prompt)
            accepted.extend(new_records)

            if progress_callback:
                progress_callback(len(accepted), total_questions, status)

    # Save JSONL
    save_jsonl(accepted, output_jsonl_path)

    # Save metadata
    metadata = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat(),
        "requested_count": total_questions,
        "accepted_count": len(accepted),
        "duplicate_count": total_duplicates,
        "rejected_count": total_rejected,
        "parse_errors": total_parse_errors[:100],  # cap for storage
        "chunks_used": len(chunk_queue),
        "question_types": question_types,
        "difficulty": difficulty,
        "custom_instructions": custom_instructions,
        "jsonl_path": str(output_jsonl_path),
    }
    pathlib.Path(output_metadata_path).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "accepted": len(accepted),
        "duplicates": total_duplicates,
        "rejected": total_rejected,
        "parse_errors": total_parse_errors,
        "jsonl_path": str(output_jsonl_path),
        "metadata_path": str(output_metadata_path),
    }
