"""
src/document_processing/docx_processor.py
──────────────────────────────────────────
Extract text from DOCX files using python-docx.
"""

import pathlib, re
from src.document_processing.pdf_processor import ExtractionResult


def extract_docx(file_path: str | pathlib.Path) -> ExtractionResult:
    path = pathlib.Path(file_path)
    try:
        from docx import Document
    except ImportError:
        return ExtractionResult(
            filename=path.name, text="", page_count=0, word_count=0,
            error="python-docx is not installed. Run: pip install python-docx"
        )
    try:
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        raw = "\n\n".join(paragraphs)
        cleaned = re.sub(r"[ \t]+", " ", raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

        if not cleaned:
            return ExtractionResult(
                filename=path.name, text="", page_count=1, word_count=0,
                error="Document is empty."
            )

        return ExtractionResult(
            filename=path.name,
            text=cleaned,
            page_count=len(doc.sections),
            word_count=len(cleaned.split()),
        )
    except Exception as exc:
        return ExtractionResult(
            filename=path.name, text="", page_count=0, word_count=0,
            error=str(exc)
        )
