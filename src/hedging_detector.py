"""
Hedging language detector for the Narrative Forensics Tool.

Detects linguistic uncertainty markers in text using:
  1. Dictionary matching — a curated list of hedging words and phrases
  2. POS-tag patterns   — modal verbs (may, might, could, would, should)
     using spaCy when available

Output: HedgingResult with per-sentence flags and a Hedging Index
        (percentage of sentences that contain at least one hedge).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hedging lexicon
# ---------------------------------------------------------------------------

# Single-word hedges (matched as whole words, case-insensitive)
HEDGE_WORDS = {
    # Epistemic adverbs
    "allegedly", "apparently", "arguably", "conceivably", "generally",
    "largely", "likely", "mainly", "mostly", "often", "ostensibly",
    "perhaps", "plausibly", "possibly", "presumably", "probably",
    "purportedly", "putatively", "reportedly", "seemingly", "supposedly",
    "typically", "usually",
    # Modal verbs
    "may", "might", "could", "would", "should", "ought",
    # Epistemic adjectives
    "alleged", "apparent", "arguable", "assumed", "believed",
    "claimed", "conjectured", "debated", "disputed", "doubtful",
    "dubious", "estimated", "hypothetical", "implied", "interpreted",
    "possible", "potential", "presumed", "probable", "proposed",
    "questionable", "rumoured", "speculative", "supposed", "suspected",
    "tentative", "theoretical", "uncertain", "unclear", "unconfirmed",
    "unproven", "unverified",
}

# Multi-word hedging phrases (matched as substrings, case-insensitive)
HEDGE_PHRASES = [
    "it is alleged",
    "it is argued",
    "it is assumed",
    "it is believed",
    "it is claimed",
    "it is considered",
    "it is contended",
    "it is debated",
    "it is estimated",
    "it is generally accepted",
    "it is held",
    "it is hypothesised",
    "it is hypothesized",
    "it is implied",
    "it is maintained",
    "it is often said",
    "it is possible",
    "it is probable",
    "it is proposed",
    "it is purported",
    "it is reported",
    "it is said",
    "it is speculated",
    "it is suggested",
    "it is supposed",
    "it is thought",
    "it is unclear",
    "it is unknown",
    "it is widely believed",
    "it is widely held",
    "it is widely thought",
    "some argue",
    "some believe",
    "some claim",
    "some consider",
    "some historians argue",
    "some historians believe",
    "some historians claim",
    "some historians suggest",
    "some scholars argue",
    "some scholars believe",
    "some scholars claim",
    "some scholars suggest",
    "some sources claim",
    "some sources suggest",
    "some suggest",
    "according to some",
    "according to certain",
    "interpreted as",
    "interpreted by some as",
    "regarded by some as",
    "seen by some as",
    "considered by some to be",
    "thought by some to be",
    "believed by some to be",
    "has been argued",
    "has been claimed",
    "has been suggested",
    "have been argued",
    "have been claimed",
    "have been suggested",
    "may have been",
    "might have been",
    "could have been",
    "would have been",
    "is thought to",
    "are thought to",
    "was thought to",
    "were thought to",
    "is believed to",
    "are believed to",
    "was believed to",
    "were believed to",
    "remains unclear",
    "remains uncertain",
    "remains unknown",
    "remains disputed",
    "remains controversial",
    "is disputed",
    "is debated",
    "is contested",
    "is controversial",
    "is uncertain",
    "is unknown",
    "is not certain",
    "not entirely clear",
    "not fully understood",
    "little is known",
    "much remains unknown",
]

# Pre-compile phrase patterns for speed
_PHRASE_PATTERNS = [
    re.compile(re.escape(p), re.IGNORECASE)
    for p in sorted(HEDGE_PHRASES, key=len, reverse=True)  # longest first
]

# Word-boundary pattern for single-word hedges
_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(HEDGE_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HedgedSentence:
    sentence: str
    sentence_index: int
    matched_terms: List[str]        # the actual hedge words/phrases detected


@dataclass
class HedgingResult:
    source_id: str
    hedged_sentences: List[HedgedSentence] = field(default_factory=list)
    all_terms: List[str] = field(default_factory=list)   # flat list of every match
    total_sentences: int = 0

    @property
    def hedged_count(self) -> int:
        return len(self.hedged_sentences)

    @property
    def hedging_index(self) -> float:
        """Fraction of sentences containing at least one hedge (0.0 – 1.0)."""
        if self.total_sentences == 0:
            return 0.0
        return self.hedged_count / self.total_sentences

    @property
    def hedging_index_pct(self) -> float:
        return round(self.hedging_index * 100, 2)

    def top_terms(self, n: int = 10) -> List[Tuple[str, int]]:
        """Return the N most frequent hedge terms as (term, count) tuples."""
        from collections import Counter
        counts = Counter(t.lower() for t in self.all_terms)
        return counts.most_common(n)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "total_sentences": self.total_sentences,
            "hedged_sentences": self.hedged_count,
            "hedging_index": round(self.hedging_index, 4),
            "hedging_index_pct": self.hedging_index_pct,
            "top_terms": [{"term": t, "count": c} for t, c in self.top_terms(20)],
            "hedged_sentence_samples": [
                {
                    "sentence": h.sentence[:200],
                    "terms": h.matched_terms,
                    "index": h.sentence_index,
                }
                for h in self.hedged_sentences[:20]
            ],
        }


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

def _find_hedges_in_sentence(sentence: str) -> List[str]:
    """
    Return all hedge terms found in a sentence.

    Checks multi-word phrases first (longest match priority), then
    single-word hedges that weren't already covered.
    """
    found: List[str] = []
    covered_spans: List[Tuple[int, int]] = []

    # 1. Multi-word phrases
    for pattern in _PHRASE_PATTERNS:
        for m in pattern.finditer(sentence):
            covered_spans.append((m.start(), m.end()))
            found.append(m.group(0).lower())

    # 2. Single-word hedges (skip if inside an already-matched phrase span)
    for m in _WORD_PATTERN.finditer(sentence):
        overlaps = any(s <= m.start() < e for s, e in covered_spans)
        if not overlaps:
            found.append(m.group(0).lower())

    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_hedging(sentences: List[str], source_id: str = "") -> HedgingResult:
    """
    Detect hedging language across a list of sentences.

    Args:
        sentences:  Pre-split list of plain-text sentences.
        source_id:  Identifier for the source document.

    Returns:
        HedgingResult with per-sentence detail and aggregate Hedging Index.
    """
    result = HedgingResult(source_id=source_id, total_sentences=len(sentences))

    for idx, sentence in enumerate(sentences):
        terms = _find_hedges_in_sentence(sentence)
        if terms:
            result.hedged_sentences.append(
                HedgedSentence(
                    sentence=sentence,
                    sentence_index=idx,
                    matched_terms=terms,
                )
            )
            result.all_terms.extend(terms)

    return result


def compare_hedging(result_a: HedgingResult, result_b: HedgingResult) -> dict:
    """
    Compare hedging levels between two versions of a document.

    Returns a dict with delta values and a direction label.
    """
    delta_index = result_b.hedging_index - result_a.hedging_index
    delta_count = result_b.hedged_count - result_a.hedged_count

    if delta_index > 0.05:
        direction = "increased"
    elif delta_index < -0.05:
        direction = "decreased"
    else:
        direction = "stable"

    terms_a = set(t for t, _ in result_a.top_terms(50))
    terms_b = set(t for t, _ in result_b.top_terms(50))

    return {
        "hedging_index_a": result_a.hedging_index_pct,
        "hedging_index_b": result_b.hedging_index_pct,
        "delta_pct": round(delta_index * 100, 2),
        "delta_count": delta_count,
        "direction": direction,
        "new_terms": sorted(terms_b - terms_a),
        "removed_terms": sorted(terms_a - terms_b),
    }
