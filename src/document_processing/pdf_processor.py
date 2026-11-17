"""
src/document_processing/pdf_processor.py
─────────────────────────────────────────
Extract text from PDF files using pdfplumber (already a project dep).
"""

import pathlib
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    filename: str
    text: str
    page_count: int
    word_count: int
    error: str = ""
    is_scanned: bool = False


def extract_pdf(file_path: str | pathlib.Path) -> ExtractionResult:
    """
    Extract and normalise text from a PDF file.
    Tries pdfplumber first (better table/layout handling).
    Falls back to PyMuPDF (fitz) for corrupt/non-standard PDFs.
    """
    import re
    path = pathlib.Path(file_path)

    # ── Attempt 1: pdfplumber ────────────────────────────────
    try:
        import pdfplumber
        pages_text: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text.strip())

        if pages_text:
            raw = "\n\n".join(pages_text)
            cleaned = re.sub(r"[ \t]+", " ", raw)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            return ExtractionResult(
                filename=path.name,
                text=cleaned,
                page_count=page_count,
                word_count=len(cleaned.split()),
            )
        # pdfplumber succeeded structurally but found no text — fall through to fitz
    except Exception:
        pass

    # ── Attempt 2: PyMuPDF (fitz) ────────────────────────────
    try:
        import fitz  # PyMuPDF
        pages_text = []
        with fitz.open(str(path)) as doc:
            page_count = len(doc)
            for page in doc:
                text = page.get_text("text") or ""
                if text.strip():
                    pages_text.append(text.strip())

        if pages_text:
            raw = "\n\n".join(pages_text)
            cleaned = re.sub(r"[ \t]+", " ", raw)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            return ExtractionResult(
                filename=path.name,
                text=cleaned,
                page_count=page_count,
                word_count=len(cleaned.split()),
            )

        # Both extractors found structure but no text → likely scanned
        return ExtractionResult(
            filename=path.name,
            text="",
            page_count=page_count,
            word_count=0,
            is_scanned=True,
            error=(
                "No extractable text found. This PDF may be scanned/image-based. "
                "OCR is required to process it."
            ),
        )

    except Exception as exc:
        return ExtractionResult(
            filename=path.name,
            text="",
            page_count=0,
            word_count=0,
            error=f"PDF extraction failed (tried pdfplumber + PyMuPDF): {exc}",
        )

