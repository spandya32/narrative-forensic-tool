"""
PDF text extraction module for the Narrative Forensics Tool.

Extracts raw text from PDF files using pdfminer.six (primary) with
a pymupdf fallback. Returns structured output compatible with the
analysis pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    page_number: int
    text: str


@dataclass
class ExtractedDocument:
    file_path: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)
    full_text: str = ""
    extraction_method: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "extraction_method": self.extraction_method,
            "full_text": self.full_text,
            "pages": [{"page_number": p.page_number, "text": p.text} for p in self.pages],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Primary extractor: pdfminer.six
# ---------------------------------------------------------------------------

def _extract_with_pdfminer(file_path: str) -> ExtractedDocument:
    """Extract text page-by-page using pdfminer.six."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages: List[PageContent] = []
    full_text_parts: List[str] = []

    for page_num, page_layout in enumerate(extract_pages(file_path), start=1):
        page_text_parts = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text_parts.append(element.get_text())
        page_text = "".join(page_text_parts).strip()
        pages.append(PageContent(page_number=page_num, text=page_text))
        if page_text:
            full_text_parts.append(page_text)

    full_text = "\n\n".join(full_text_parts)
    return ExtractedDocument(
        file_path=file_path,
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
        extraction_method="pdfminer.six",
    )


# ---------------------------------------------------------------------------
# Fallback extractor: pymupdf (fitz)
# ---------------------------------------------------------------------------

def _extract_with_pymupdf(file_path: str) -> ExtractedDocument:
    """Extract text page-by-page using pymupdf (fitz)."""
    import fitz  # type: ignore

    doc = fitz.open(file_path)
    pages: List[PageContent] = []
    full_text_parts: List[str] = []

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        pages.append(PageContent(page_number=page_num, text=page_text))
        if page_text:
            full_text_parts.append(page_text)

    doc.close()
    full_text = "\n\n".join(full_text_parts)
    return ExtractedDocument(
        file_path=file_path,
        total_pages=len(pages),
        pages=pages,
        full_text=full_text,
        extraction_method="pymupdf",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pdf(file_path: str) -> ExtractedDocument:
    """
    Extract text from a PDF file.

    Tries pdfminer.six first; falls back to pymupdf if unavailable or
    if extraction fails.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        ExtractedDocument with page-level text and concatenated full text.
    """
    if not os.path.isfile(file_path):
        return ExtractedDocument(
            file_path=file_path,
            total_pages=0,
            error=f"File not found: {file_path}",
        )

    # Try pdfminer.six first
    try:
        return _extract_with_pdfminer(file_path)
    except ImportError:
        pass
    except Exception as exc:
        pdfminer_error = str(exc)
    else:
        pdfminer_error = None

    # Fallback to pymupdf
    try:
        return _extract_with_pymupdf(file_path)
    except ImportError:
        return ExtractedDocument(
            file_path=file_path,
            total_pages=0,
            error=(
                "Neither pdfminer.six nor pymupdf is installed. "
                "Install one with: pip install pdfminer.six   or   pip install pymupdf"
            ),
        )
    except Exception as exc:
        return ExtractedDocument(
            file_path=file_path,
            total_pages=0,
            error=f"Extraction failed: {exc}",
        )
