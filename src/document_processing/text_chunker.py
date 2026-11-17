"""
src/document_processing/text_chunker.py
────────────────────────────────────────
Split long document text into overlapping chunks suitable for API calls.
Chunks respect sentence/paragraph boundaries where possible.
"""

import re
from dataclasses import dataclass
from src.config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    text: str
    source: str          # original filename
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int


def _approx_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return len(text) // 4


def chunk_text(
    text: str,
    source: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """
    Split text into chunks of at most max_tokens (approx).
    Adjacent chunks overlap by overlap_chars to preserve context.
    Splits on paragraph boundaries first, then sentence, then character.
    """
    max_chars = max_tokens * 4

    # Split into paragraphs
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Para itself might be too long — split by sentence
            if len(para) > max_chars:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sent_buf = ""
                for sent in sentences:
                    cand2 = (sent_buf + " " + sent).strip() if sent_buf else sent
                    if len(cand2) <= max_chars:
                        sent_buf = cand2
                    else:
                        if sent_buf:
                            chunks.append(sent_buf)
                        # Hard cut if a single sentence exceeds limit
                        if len(sent) > max_chars:
                            for start in range(0, len(sent), max_chars - overlap_chars):
                                chunks.append(sent[start:start + max_chars])
                            sent_buf = ""
                        else:
                            sent_buf = sent
                if sent_buf:
                    current = sent_buf
                else:
                    current = ""
            else:
                current = para.strip()

    if current:
        chunks.append(current)

    # Add overlap
    result: list[Chunk] = []
    pos = 0
    for i, chunk in enumerate(chunks):
        # Find char start in original text
        start = text.find(chunk[:min(50, len(chunk))], pos)
        end = start + len(chunk) if start >= 0 else pos + len(chunk)
        # Prepend tail of previous chunk for overlap
        if i > 0 and overlap_chars > 0:
            overlap = chunks[i - 1][-overlap_chars:]
            full_text = overlap + "\n\n" + chunk
        else:
            full_text = chunk
        result.append(Chunk(
            text=full_text,
            source=source,
            chunk_index=i,
            total_chunks=len(chunks),
            char_start=start if start >= 0 else 0,
            char_end=end,
        ))
        pos = max(0, end - overlap_chars)

    return result


def chunk_documents(
    documents: list[dict],  # [{"filename": str, "text": str}, ...]
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk all documents and return a flat list of chunks."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_text(
            doc["text"], doc["filename"], max_tokens, overlap_chars
        )
        all_chunks.extend(chunks)
    return all_chunks
