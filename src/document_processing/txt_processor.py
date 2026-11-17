"""
src/document_processing/txt_processor.py
─────────────────────────────────────────
Extract text from plain-text files (UTF-8, latin-1 fallback).
"""

import pathlib, re
from src.document_processing.pdf_processor import ExtractionResult


def extract_txt(file_path: str | pathlib.Path) -> ExtractionResult:
    path = pathlib.Path(file_path)
    try:
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                raw = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            return ExtractionResult(
                filename=path.name, text="", page_count=0, word_count=0,
                error="Could not decode file with UTF-8, Latin-1, or CP1252."
            )
    except Exception as exc:
        return ExtractionResult(
            filename=path.name, text="", page_count=0, word_count=0,
            error=str(exc)
        )

    cleaned = re.sub(r"[ \t]+", " ", raw)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if not cleaned:
        return ExtractionResult(
            filename=path.name, text="", page_count=1, word_count=0,
            error="File is empty."
        )

    return ExtractionResult(
        filename=path.name,
        text=cleaned,
        page_count=1,
        word_count=len(cleaned.split()),
    )
