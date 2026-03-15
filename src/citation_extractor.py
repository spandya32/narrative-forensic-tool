"""
Citation extraction module for the Narrative Forensics Tool.

Detects and classifies references in plain text:
  - Inline numeric references       [1], [23]
  - Author-year references          (Smith, 2001); Smith (2001)
  - Footnote / endnote markers      ¹ ² ³ or superscript digits
  - Bare URLs                       https://... / http://...
  - DOIs                            doi:10.XXXX/...
  - Book / chapter references       heuristic title-case phrases following "in"/"see"
  - Wikipedia <ref> tags            already cleaned to plain text by fetcher

Outputs:
  - List of Citation objects with type, raw text, and position
  - Source persistence score when comparing two citation sets
  - Added / removed citations between two versions
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    citation_type: str      # "numeric" | "author_year" | "url" | "doi" | "footnote" | "narrative"
    raw_text: str           # the matched string
    sentence: str           # sentence it was found in
    sentence_index: int     # index in the sentence list
    author: str = ""        # extracted author if detectable
    year: str = ""          # extracted year if detectable
    url: str = ""           # URL if type == "url"

    def to_dict(self) -> dict:
        return {
            "type": self.citation_type,
            "raw": self.raw_text,
            "author": self.author,
            "year": self.year,
            "url": self.url,
            "sentence_index": self.sentence_index,
        }


@dataclass
class CitationReport:
    source_id: str
    citations: List[Citation] = field(default_factory=list)
    # Comparison fields (populated by compare_citations)
    added: List[Citation] = field(default_factory=list)
    removed: List[Citation] = field(default_factory=list)
    persistence_score: float = 0.0   # fraction of A's citations still in B

    @property
    def citation_count(self) -> int:
        return len(self.citations)

    @property
    def unique_authors(self) -> List[str]:
        return sorted({c.author for c in self.citations if c.author})

    @property
    def unique_urls(self) -> List[str]:
        return sorted({c.url for c in self.citations if c.url})

    def by_type(self) -> dict:
        counts: dict = {}
        for c in self.citations:
            counts[c.citation_type] = counts.get(c.citation_type, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "citation_count": self.citation_count,
            "unique_authors": self.unique_authors,
            "by_type": self.by_type(),
            "persistence_score": round(self.persistence_score, 4),
            "added_count": len(self.added),
            "removed_count": len(self.removed),
            "citations": [c.to_dict() for c in self.citations[:100]],
        }


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# [1], [23], [1,2], [1-3]
_NUMERIC_REF = re.compile(r"\[(\d+(?:[,\-–]\d+)*)\]")

# (Smith, 2001) or (Smith and Jones, 2001) or (Smith et al., 2001)
_AUTHOR_YEAR_PAREN = re.compile(
    r"\(([A-Z][a-zA-Zé\-']+(?: (?:and|&|et al\.?) [A-Z][a-zA-Zé\-']+)?),?\s*(\d{4}[a-z]?)\)"
)

# Smith (2001) — author name followed by parenthesised year
_AUTHOR_YEAR_INLINE = re.compile(
    r"\b([A-Z][a-zA-Zé\-']{2,}(?:\s+[A-Z][a-zA-Zé\-']{2,})?)\s+\((\d{4}[a-z]?)\)"
)

# Bare URLs
_URL = re.compile(r"https?://[^\s\)>\"']{4,}", re.IGNORECASE)

# DOIs
_DOI = re.compile(r"\bdoi\s*:\s*10\.\d{4,}/\S+", re.IGNORECASE)

# Unicode superscript digits or footnote markers  ¹²³ or ^1 ^2
_FOOTNOTE = re.compile(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]+|\^(\d+)")

# Narrative references: "according to X", "cited by X", "as noted in X"
_NARRATIVE_REF = re.compile(
    r"\b(?:according to|cited (?:in|by)|as (?:noted|stated|argued|claimed) (?:in|by)|"
    r"see also|cf\.|op\. cit\.|ibid\.?)\b",
    re.IGNORECASE,
)

# ibid / op. cit. standalone
_IBID = re.compile(r"\b(ibid\.?|op\.?\s*cit\.?)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_year(text: str) -> str:
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", text)
    return m.group(1) if m else ""


def _normalise_citation_key(c: Citation) -> str:
    """Produce a normalised string key for deduplication and comparison."""
    if c.citation_type == "url":
        return f"url::{c.url.rstrip('/')}"
    if c.citation_type == "doi":
        return f"doi::{c.raw_text.lower()}"
    if c.author and c.year:
        return f"authoryear::{c.author.lower()}::{c.year}"
    if c.citation_type == "numeric":
        return f"numeric::{c.raw_text}"
    return f"raw::{c.raw_text.lower().strip()}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_citations(sentences: List[str], source_id: str = "") -> CitationReport:
    """
    Extract all citations from a list of sentences.

    Args:
        sentences:  Pre-split list of plain-text sentences.
        source_id:  Label for the source document.

    Returns:
        CitationReport with all detected Citation objects.
    """
    report = CitationReport(source_id=source_id)

    for idx, sentence in enumerate(sentences):
        _extract_from_sentence(sentence, idx, report.citations)

    # Also scan for bibliography-style entries (end-of-book reference lists)
    _extract_bibliography_citations(sentences, report.citations)

    return report


# Pattern for bibliography / reference-list entries:
# "Author, A. (2001). Title..." or "Author, A. 2001. Title..."
_BIBLIO_ENTRY = re.compile(
    r"^[A-Z][a-zA-Zé\-']{1,30},\s+[A-Z][\w\.\s]{0,30}"  # Author, Initials
    r"[\.\(]\s*(1[5-9]\d{2}|20\d{2})[\.\)]",              # year
    re.MULTILINE,
)

# Chapter-wise bibliography marker (Sharma style: "Chapter 1\nAuthor (year)")
_BIBLIO_SECTION = re.compile(
    r"\b(bibliography|references|further reading|select bibliography|"
    r"bibliographical note|works cited|notes and references)\b",
    re.IGNORECASE,
)


def _extract_bibliography_citations(sentences: List[str], out: List[Citation]) -> None:
    """
    Detect bibliography/reference-list sections and count their entries.

    Adds one Citation per detected bibliographic entry found in the
    reference section at the end of the document.
    """
    # Find the sentence index where the bibliography section begins
    biblio_start = None
    for idx in range(len(sentences) - 1, max(len(sentences) - 500, -1), -1):
        if _BIBLIO_SECTION.search(sentences[idx]):
            biblio_start = idx
            break

    if biblio_start is None:
        return

    # In the bibliography section, look for individual citation entries
    for idx in range(biblio_start + 1, len(sentences)):
        sentence = sentences[idx]
        if _BIBLIO_ENTRY.match(sentence.strip()):
            m = re.search(r"(1[5-9]\d{2}|20\d{2})", sentence)
            author_m = re.match(r"([A-Z][a-zA-Zé\-']{1,30})", sentence.strip())
            out.append(Citation(
                citation_type="bibliography",
                raw_text=sentence[:120],
                sentence=sentence,
                sentence_index=idx,
                author=author_m.group(1) if author_m else "",
                year=m.group(1) if m else "",
            ))


def _extract_from_sentence(sentence: str, idx: int, out: List[Citation]) -> None:
    """Detect all citation types within a single sentence."""

    # URLs (check first so we don't double-match inside URLs)
    for m in _URL.finditer(sentence):
        out.append(Citation(
            citation_type="url",
            raw_text=m.group(0),
            sentence=sentence,
            sentence_index=idx,
            url=m.group(0),
        ))

    # DOIs
    for m in _DOI.finditer(sentence):
        out.append(Citation(
            citation_type="doi",
            raw_text=m.group(0),
            sentence=sentence,
            sentence_index=idx,
        ))

    # Author-year parenthetical  (Smith, 2001)
    for m in _AUTHOR_YEAR_PAREN.finditer(sentence):
        out.append(Citation(
            citation_type="author_year",
            raw_text=m.group(0),
            sentence=sentence,
            sentence_index=idx,
            author=m.group(1).strip(),
            year=m.group(2),
        ))

    # Author-year inline  Smith (2001) — only if not already caught above
    paren_spans = {m.span() for m in _AUTHOR_YEAR_PAREN.finditer(sentence)}
    for m in _AUTHOR_YEAR_INLINE.finditer(sentence):
        # Skip if this overlaps with a parenthetical match
        if any(m.start() >= s and m.end() <= e for s, e in paren_spans):
            continue
        out.append(Citation(
            citation_type="author_year",
            raw_text=m.group(0),
            sentence=sentence,
            sentence_index=idx,
            author=m.group(1).strip(),
            year=m.group(2),
        ))

    # Numeric [1]
    for m in _NUMERIC_REF.finditer(sentence):
        out.append(Citation(
            citation_type="numeric",
            raw_text=m.group(0),
            sentence=sentence,
            sentence_index=idx,
        ))

    # Footnote superscripts
    for m in _FOOTNOTE.finditer(sentence):
        out.append(Citation(
            citation_type="footnote",
            raw_text=m.group(0),
            sentence=sentence,
            sentence_index=idx,
        ))

    # Narrative references ("according to", "cited by", …)
    for m in _NARRATIVE_REF.finditer(sentence):
        out.append(Citation(
            citation_type="narrative",
            raw_text=sentence[m.start():min(m.end() + 60, len(sentence))],
            sentence=sentence,
            sentence_index=idx,
        ))


def compare_citations(report_a: CitationReport, report_b: CitationReport) -> None:
    """
    Populate report_b.added, report_a.removed, and report_a.persistence_score
    by comparing the citation sets of two versions of the same document.

    Modifies report_a and report_b in-place.
    """
    keys_a: Set[str] = {_normalise_citation_key(c) for c in report_a.citations}
    keys_b: Set[str] = {_normalise_citation_key(c) for c in report_b.citations}

    added_keys = keys_b - keys_a
    removed_keys = keys_a - keys_b

    report_b.added = [c for c in report_b.citations if _normalise_citation_key(c) in added_keys]
    report_a.removed = [c for c in report_a.citations if _normalise_citation_key(c) in removed_keys]

    if keys_a:
        report_a.persistence_score = len(keys_a & keys_b) / len(keys_a)
    else:
        report_a.persistence_score = 1.0
