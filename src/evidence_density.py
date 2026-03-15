"""
Evidence density calculator for the Narrative Forensics Tool.

Formula (from CLAUDE.md):
    Evidence Density = citation sentences / claim sentences

A "claim sentence" is one that makes an assertive proposition.
A "citation sentence" is one that includes a reference marker.

Both classifiers are inherited from text_preprocessor (Phase 1).
This module adds:
  - finer-grained claim scoring (strong vs. weak claims)
  - per-section evidence density when section data is available
  - change detection between two versions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SectionDensity:
    section_title: str
    claim_count: int
    citation_count: int
    density: float


@dataclass
class EvidenceDensityResult:
    source_id: str
    total_sentences: int
    claim_count: int
    citation_count: int
    density: float                          # citations / claims (or 0 if no claims)
    section_densities: List[SectionDensity] = field(default_factory=list)

    @property
    def density_label(self) -> str:
        if self.density >= 0.75:
            return "strong"
        if self.density >= 0.50:
            return "moderate"
        if self.density >= 0.25:
            return "weak"
        return "poor"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "total_sentences": self.total_sentences,
            "claim_count": self.claim_count,
            "citation_count": self.citation_count,
            "density": round(self.density, 4),
            "density_label": self.density_label,
            "section_densities": [
                {
                    "section": s.section_title,
                    "claims": s.claim_count,
                    "citations": s.citation_count,
                    "density": round(s.density, 4),
                }
                for s in self.section_densities
            ],
        }

    def summary(self) -> str:
        return (
            f"Claims     : {self.claim_count}\n"
            f"Citations  : {self.citation_count}\n"
            f"Density    : {self.density:.4f}  ({self.density_label})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_density(preprocessed, source_id: str = "") -> EvidenceDensityResult:
    """
    Calculate evidence density from a PreprocessedDocument (Phase 1 output).

    Args:
        preprocessed: A text_preprocessor.PreprocessedDocument instance.
        source_id:    Label override (defaults to preprocessed.source_id).

    Returns:
        EvidenceDensityResult.
    """
    sid = source_id or preprocessed.source_id
    claim_count = preprocessed.claim_count
    citation_count = preprocessed.citation_count
    density = citation_count / claim_count if claim_count > 0 else 0.0

    return EvidenceDensityResult(
        source_id=sid,
        total_sentences=preprocessed.sentence_count,
        claim_count=claim_count,
        citation_count=citation_count,
        density=density,
    )


def calculate_density_from_sentences(
    sentences: List[str],
    source_id: str = "",
) -> EvidenceDensityResult:
    """
    Calculate evidence density directly from a list of sentences without
    requiring a full PreprocessedDocument.

    Runs the claim and citation classifiers inline.

    Args:
        sentences:  List of plain-text sentences.
        source_id:  Label for the source document.

    Returns:
        EvidenceDensityResult.
    """
    # Import classifiers from text_preprocessor to avoid duplication
    from text_preprocessor import _is_claim_sentence, _is_citation_sentence

    claim_indices = []
    citation_indices = []

    for idx, sent in enumerate(sentences):
        if _is_claim_sentence(sent):
            claim_indices.append(idx)
        if _is_citation_sentence(sent):
            citation_indices.append(idx)

    claim_count = len(claim_indices)
    citation_count = len(citation_indices)
    density = citation_count / claim_count if claim_count > 0 else 0.0

    return EvidenceDensityResult(
        source_id=source_id,
        total_sentences=len(sentences),
        claim_count=claim_count,
        citation_count=citation_count,
        density=density,
    )


def calculate_section_densities(
    sections: List[dict],
    source_id: str = "",
) -> List[SectionDensity]:
    """
    Compute per-section evidence density for a document with known sections.

    Args:
        sections:  List of {title, text} dicts (from wikipedia_fetcher or
                   a manually segmented PDF).
        source_id: Label for the source.

    Returns:
        List of SectionDensity objects.
    """
    from text_preprocessor import _is_claim_sentence, _is_citation_sentence
    import re

    results: List[SectionDensity] = []
    sentence_split = re.compile(r"(?<=[.!?])\s+")

    for section in sections:
        title = section.get("title", "Untitled")
        text = section.get("text", "")
        if not text.strip():
            continue

        sents = [s.strip() for s in sentence_split.split(text) if s.strip()]
        claims = sum(1 for s in sents if _is_claim_sentence(s))
        citations = sum(1 for s in sents if _is_citation_sentence(s))
        density = citations / claims if claims > 0 else 0.0

        results.append(SectionDensity(
            section_title=title,
            claim_count=claims,
            citation_count=citations,
            density=density,
        ))

    return results


def compare_density(
    result_a: EvidenceDensityResult,
    result_b: EvidenceDensityResult,
) -> dict:
    """
    Compare evidence density between two versions of a document.

    Returns a dict with delta values and an interpretation.
    """
    delta = result_b.density - result_a.density

    if delta < -0.10:
        trend = "declining — citations reduced relative to claims"
    elif delta > 0.10:
        trend = "improving — more claims now have citation support"
    else:
        trend = "stable"

    return {
        "density_a": round(result_a.density, 4),
        "density_b": round(result_b.density, 4),
        "delta": round(delta, 4),
        "trend": trend,
        "label_a": result_a.density_label,
        "label_b": result_b.density_label,
    }
