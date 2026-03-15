"""
Narrative compression detector for the Narrative Forensics Tool.

Detects oversimplification of complex events (CLAUDE.md §10).

Metrics:
  - actors_count         : unique named actors (PERSON + ORG)
  - causal_claims_count  : sentences containing causal language
  - events_count         : sentences containing event-marker verbs
  - compression_ratio    : actors_count / causal_claims_count

Low compression ratio → few actors account for many causal claims
                         (possible narrative oversimplification).

Also computes an overall Compression Score (0–1) that normalises
these metrics relative to document length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Causal language patterns
# ---------------------------------------------------------------------------

_CAUSAL_PATTERNS = re.compile(
    r"\b("
    r"caused|cause[sd]?\s+by|led\s+to|result(?:ed|s)?\s+(?:in|from)|"
    r"trigger(?:ed|s)?|brought\s+about|gave\s+rise\s+to|"
    r"because\s+of|due\s+to|owing\s+to|on\s+account\s+of|"
    r"consequently|therefore|thus|hence|as\s+a\s+(?:result|consequence)|"
    r"stem(?:s|med)?\s+from|arose?\s+(?:from|out\s+of)|"
    r"responsible\s+for|attributed\s+to|driven\s+by|prompted\s+by"
    r")\b",
    re.IGNORECASE,
)

# Event-marker verbs (things that happened)
_EVENT_PATTERNS = re.compile(
    r"\b("
    r"invaded|conquered|defeated|captured|destroyed|founded|established|"
    r"built|constructed|ruled|governed|declared|signed|issued|passed|"
    r"launched|began|started|ended|ceased|fell|rose|collapsed|emerged|"
    r"occurred|happened|took\s+place|was\s+(?:founded|established|built|"
    r"created|formed|dissolved|defeated|captured|destroyed|killed|executed)"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    source_id: str
    actors_count: int
    causal_claims_count: int
    events_count: int
    compression_ratio: float        # actors / causal_claims (or 0)
    total_sentences: int

    # Normalised scores (per 100 sentences)
    actors_per_100: float = 0.0
    causal_per_100: float = 0.0
    events_per_100: float = 0.0

    causal_sentence_indices: List[int] = field(default_factory=list)
    event_sentence_indices: List[int] = field(default_factory=list)
    actor_list: List[str] = field(default_factory=list)

    @property
    def compression_label(self) -> str:
        if self.compression_ratio == 0:
            return "indeterminate"
        if self.compression_ratio < 0.5:
            return "high compression (few actors, many causal claims)"
        if self.compression_ratio < 1.5:
            return "moderate compression"
        return "low compression (diverse actors)"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "actors_count": self.actors_count,
            "causal_claims_count": self.causal_claims_count,
            "events_count": self.events_count,
            "compression_ratio": round(self.compression_ratio, 4),
            "compression_label": self.compression_label,
            "actors_per_100_sentences": round(self.actors_per_100, 2),
            "causal_per_100_sentences": round(self.causal_per_100, 2),
            "events_per_100_sentences": round(self.events_per_100, 2),
            "actor_list": self.actor_list[:30],
        }

    def summary(self) -> str:
        return (
            f"Actors detected       : {self.actors_count}\n"
            f"Causal claims         : {self.causal_claims_count}\n"
            f"Events detected       : {self.events_count}\n"
            f"Compression Ratio     : {self.compression_ratio:.4f}\n"
            f"                        ({self.compression_label})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_compression(preprocessed, source_id: str = "") -> CompressionResult:
    """
    Detect narrative compression in a PreprocessedDocument.

    Args:
        preprocessed: PreprocessedDocument from text_preprocessor.
        source_id:    Label override.

    Returns:
        CompressionResult.
    """
    from text_preprocessor import extract_unique_entities

    sid = source_id or preprocessed.source_id
    sents = preprocessed.sentences
    total = len(sents)

    # Actors: unique PERSON + ORG entities
    actors = extract_unique_entities(preprocessed, labels=["PERSON", "ORG"])
    actor_names = [a["text"] for a in actors]
    actors_count = len(actor_names)

    # Causal claims: sentences containing causal language
    causal_indices = [
        i for i, s in enumerate(sents)
        if _CAUSAL_PATTERNS.search(s)
    ]
    causal_count = len(causal_indices)

    # Events: sentences with event-marker verbs
    event_indices = [
        i for i, s in enumerate(sents)
        if _EVENT_PATTERNS.search(s)
    ]
    events_count = len(event_indices)

    ratio = actors_count / causal_count if causal_count > 0 else 0.0

    scale = 100 / total if total > 0 else 0.0

    return CompressionResult(
        source_id=sid,
        actors_count=actors_count,
        causal_claims_count=causal_count,
        events_count=events_count,
        compression_ratio=ratio,
        total_sentences=total,
        actors_per_100=actors_count * scale,
        causal_per_100=causal_count * scale,
        events_per_100=events_count * scale,
        causal_sentence_indices=causal_indices,
        event_sentence_indices=event_indices,
        actor_list=actor_names,
    )


def compare_compression(result_a: CompressionResult, result_b: CompressionResult) -> dict:
    """Compare compression metrics between two documents."""
    delta_ratio = result_b.compression_ratio - result_a.compression_ratio
    actors_only_a = set(result_a.actor_list) - set(result_b.actor_list)
    actors_only_b = set(result_b.actor_list) - set(result_a.actor_list)

    return {
        "ratio_a": round(result_a.compression_ratio, 4),
        "ratio_b": round(result_b.compression_ratio, 4),
        "delta_ratio": round(delta_ratio, 4),
        "actors_added": sorted(actors_only_b)[:20],
        "actors_removed": sorted(actors_only_a)[:20],
        "label_a": result_a.compression_label,
        "label_b": result_b.compression_label,
    }
