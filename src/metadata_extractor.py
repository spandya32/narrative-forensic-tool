"""
Document metadata extractor for the Narrative Forensics Tool.

Extracts structured metadata (title, author, year, source type) from:
  - PDF files      → PDF metadata + heuristic first-page analysis
  - Wikipedia      → MediaWiki API fields
  - Plain text     → heuristic pattern matching

Output conforms to the DATA_SCHEMA.md Document Metadata section.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    title: str = ""
    author: str = ""
    year: str = ""
    source_type: str = ""   # Book | Article | Wikipedia | Text
    file_path: str = ""
    url: str = ""
    language: str = "en"
    extra: dict = field(default_factory=dict)   # any extra fields from the source

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "year": self.year,
            "source_type": self.source_type,
            "file_path": self.file_path,
            "url": self.url,
            "language": self.language,
            **self.extra,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Document Metadata\n",
            f"Title: {self.title or 'Unknown'}",
            f"Author: {self.author or 'Unknown'}",
            f"Year: {self.year or 'Unknown'}",
            f"Source Type: {self.source_type or 'Unknown'}",
        ]
        if self.url:
            lines.append(f"URL: {self.url}")
        if self.file_path:
            lines.append(f"File: {self.file_path}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF metadata
# ---------------------------------------------------------------------------

def _year_from_string(s: str) -> str:
    """Extract a 4-digit year from a string, or return empty string."""
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", s)
    return m.group(1) if m else ""


def extract_pdf_metadata(file_path: str, first_page_text: str = "") -> DocumentMetadata:
    """
    Extract metadata from a PDF file.

    Tries to read the embedded PDF metadata first; falls back to
    heuristic scanning of the first page text for title and author.

    Args:
        file_path:       Path to the PDF file.
        first_page_text: Optional first-page plain text already extracted.

    Returns:
        DocumentMetadata populated from available sources.
    """
    meta = DocumentMetadata(file_path=file_path, source_type="Book")

    # ---- try pdfminer.six -----------------------------------------------
    _read_pdfminer_metadata(file_path, meta)

    # ---- fallback: pymupdf -----------------------------------------------
    if not meta.title and not meta.author:
        _read_pymupdf_metadata(file_path, meta)

    # ---- fallback: heuristic first-page scan -----------------------------
    if first_page_text and (not meta.title or not meta.author):
        _heuristic_firstpage(first_page_text, meta)

    # ---- derive year from creation date if still missing -----------------
    if not meta.year:
        creation = meta.extra.get("creation_date", "")
        if creation:
            meta.year = _year_from_string(creation)

    # ---- last resort: use filename as title ------------------------------
    if not meta.title:
        meta.title = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")

    return meta


def _read_pdfminer_metadata(file_path: str, meta: DocumentMetadata) -> None:
    try:
        from pdfminer.high_level import extract_pages  # noqa: F401
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument

        with open(file_path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            info = doc.info[0] if doc.info else {}
            meta.title = _decode_pdf_field(info.get("Title", ""))
            meta.author = _decode_pdf_field(info.get("Author", ""))
            meta.year = _year_from_string(_decode_pdf_field(info.get("CreationDate", "")))
            meta.extra["creation_date"] = _decode_pdf_field(info.get("CreationDate", ""))
            meta.extra["subject"] = _decode_pdf_field(info.get("Subject", ""))
    except Exception:
        pass


def _read_pymupdf_metadata(file_path: str, meta: DocumentMetadata) -> None:
    try:
        import fitz  # type: ignore

        doc = fitz.open(file_path)
        info = doc.metadata
        doc.close()
        if info:
            meta.title = meta.title or (info.get("title") or "")
            meta.author = meta.author or (info.get("author") or "")
            if not meta.year:
                creation = info.get("creationDate") or info.get("modDate") or ""
                meta.year = _year_from_string(creation)
            meta.extra["creation_date"] = meta.extra.get("creation_date") or info.get("creationDate", "")
    except Exception:
        pass


def _decode_pdf_field(value) -> str:
    """Decode a PDF metadata field which may be bytes or str."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""
    return str(value).strip() if value else ""


# Patterns to detect author lines on a title page
_AUTHOR_PATTERNS = [
    re.compile(r"^(?:by|author[s]?)[:\s]+(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$", re.MULTILINE),
]

_YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def _heuristic_firstpage(text: str, meta: DocumentMetadata) -> None:
    """Attempt to extract title and author from first-page text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if not meta.title and lines:
        # Take the longest line in the first 10 lines as a title candidate
        candidates = sorted(lines[:10], key=len, reverse=True)
        meta.title = candidates[0] if candidates else ""

    if not meta.author:
        for pattern in _AUTHOR_PATTERNS:
            m = pattern.search(text[:2000])
            if m:
                candidate = m.group(1).strip()
                # Sanity check: not too long, not all caps (which looks like a heading)
                if 3 < len(candidate) < 80 and not candidate.isupper():
                    meta.author = candidate
                    break

    if not meta.year:
        m = _YEAR_PATTERN.search(text[:3000])
        if m:
            meta.year = m.group(1)


# ---------------------------------------------------------------------------
# Wikipedia metadata
# ---------------------------------------------------------------------------

def extract_wikipedia_metadata(article) -> DocumentMetadata:
    """
    Build DocumentMetadata from a WikiArticle object returned by
    wikipedia_fetcher.fetch_article().

    Args:
        article: WikiArticle instance.

    Returns:
        DocumentMetadata with source_type = "Wikipedia".
    """
    meta = DocumentMetadata(
        title=article.title,
        source_type="Wikipedia",
        url=article.url,
    )

    # Extract year from the oldest available revision timestamp
    if article.revisions:
        oldest = article.revisions[-1]
        meta.year = _year_from_string(oldest.timestamp)
        meta.extra["oldest_revision_timestamp"] = oldest.timestamp
        meta.extra["oldest_revision_user"] = oldest.user

    meta.extra["page_id"] = article.page_id
    meta.extra["categories"] = article.categories

    return meta


# ---------------------------------------------------------------------------
# Plain text metadata (heuristic)
# ---------------------------------------------------------------------------

def extract_text_metadata(file_path: str, text: str = "") -> DocumentMetadata:
    """
    Extract metadata from a plain text file using heuristic patterns.

    Args:
        file_path: Path to the text file.
        text:      Optional pre-loaded text content.

    Returns:
        DocumentMetadata with source_type = "Text".
    """
    meta = DocumentMetadata(file_path=file_path, source_type="Text")

    if not text:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(4096)   # only scan first 4 KB
        except OSError:
            meta.title = os.path.basename(file_path)
            return meta

    _heuristic_firstpage(text, meta)

    if not meta.title:
        meta.title = os.path.splitext(os.path.basename(file_path))[0].replace("_", " ")

    return meta


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def extract_metadata(
    source_type: str,
    file_path: str = "",
    url: str = "",
    text: str = "",
    article=None,
) -> DocumentMetadata:
    """
    Unified metadata extractor.

    Args:
        source_type: One of "pdf", "wikipedia", "text".
        file_path:   Path to the file (for pdf / text sources).
        url:         URL (for wikipedia source).
        text:        Optional pre-extracted text (for pdf first-page heuristics
                     or plain text sources).
        article:     Optional WikiArticle object (for wikipedia source).

    Returns:
        DocumentMetadata.
    """
    s = source_type.lower()
    if s == "pdf":
        return extract_pdf_metadata(file_path, first_page_text=text)
    if s == "wikipedia":
        if article is not None:
            return extract_wikipedia_metadata(article)
        # Minimal fallback if no article object provided
        meta = DocumentMetadata(url=url, source_type="Wikipedia")
        return meta
    if s == "text":
        return extract_text_metadata(file_path, text=text)

    # Unknown type
    meta = DocumentMetadata(file_path=file_path, url=url, source_type=source_type)
    return meta
