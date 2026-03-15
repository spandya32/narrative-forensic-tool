"""
Sentiment analyzer for the Narrative Forensics Tool.

Goal (from CLAUDE.md): detect asymmetric skepticism toward specific
entities — i.e. whether the text applies consistently different tones
to different groups, authors, or institutions.

Pipeline:
  1. Extract named entities from the PreprocessedDocument (Phase 1 NER)
  2. Locate the sentence each entity appears in
  3. Score that sentence's sentiment using VADER (nltk) if available,
     falling back to a lightweight lexicon scorer
  4. Aggregate per-entity sentiment distributions
  5. Compute an Asymmetry Score between entity pairs

VADER is the primary scorer because it handles informal and academic
prose better than naïve word counts. It is installed via:
    pip install nltk
    python -m nltk.downloader vader_lexicon
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Fallback lexicon (used when VADER is unavailable)
# ---------------------------------------------------------------------------

# A minimal positive/negative word set for graceful degradation
_POS_WORDS = {
    "acclaimed", "accurate", "admired", "advanced", "authentic", "balanced",
    "celebrated", "clear", "comprehensive", "confirmed", "credible",
    "decisive", "documented", "established", "excellent", "factual",
    "genuine", "groundbreaking", "important", "innovative", "insightful",
    "legitimate", "major", "notable", "objective", "praised", "proven",
    "reliable", "respected", "rigorous", "scholarly", "significant",
    "sound", "thorough", "trustworthy", "valid", "valuable", "verified",
    "well-regarded", "well-supported",
}

_NEG_WORDS = {
    "alleged", "baseless", "biased", "challenged", "controversial",
    "debunked", "discredited", "disputed", "doubtful", "erroneous",
    "fabricated", "false", "flawed", "fringe", "inaccurate", "incorrect",
    "invented", "marginal", "misleading", "misrepresented", "mistaken",
    "mythical", "obscure", "outdated", "polemical", "pseudo", "questionable",
    "refuted", "rejected", "speculative", "spurious", "suspect",
    "unconfirmed", "unfounded", "unproven", "unreliable", "unsupported",
    "unverified", "wrong",
}

_WORD_RE = re.compile(r"\b\w+\b")


def _lexicon_score(text: str) -> float:
    """
    Simple lexicon-based polarity score in [-1, +1].
    Returns 0.0 for neutral or empty text.
    """
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in _POS_WORDS)
    neg = sum(1 for w in words if w in _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


# ---------------------------------------------------------------------------
# VADER loader (lazy)
# ---------------------------------------------------------------------------

_vader = None
_vader_available: Optional[bool] = None


def _get_vader():
    global _vader, _vader_available
    if _vader_available is not None:
        return _vader
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        _vader = SentimentIntensityAnalyzer()
        _vader_available = True
    except Exception:
        _vader_available = False
        _vader = None
    return _vader


def _score_sentence(sentence: str) -> float:
    """
    Return a compound polarity score in [-1, +1] for the given sentence.
    Uses VADER when available, falls back to lexicon scorer.
    """
    vader = _get_vader()
    if vader is not None:
        try:
            return vader.polarity_scores(sentence)["compound"]
        except Exception:
            pass
    return _lexicon_score(sentence)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntitySentiment:
    entity_text: str
    entity_label: str
    sentence: str
    sentence_index: int
    score: float            # compound polarity [-1, +1]


@dataclass
class EntityProfile:
    entity_text: str
    entity_label: str
    mention_count: int
    scores: List[float]
    mean_score: float
    tone: str               # "positive" | "negative" | "neutral" | "mixed"

    def to_dict(self) -> dict:
        return {
            "entity": self.entity_text,
            "label": self.entity_label,
            "mentions": self.mention_count,
            "mean_score": round(self.mean_score, 4),
            "tone": self.tone,
        }


@dataclass
class SentimentResult:
    source_id: str
    scorer: str                                         # "vader" or "lexicon"
    entity_sentiments: List[EntitySentiment] = field(default_factory=list)
    entity_profiles: List[EntityProfile] = field(default_factory=list)
    asymmetry_pairs: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "scorer": self.scorer,
            "entity_count": len(self.entity_profiles),
            "entity_profiles": [p.to_dict() for p in self.entity_profiles],
            "asymmetry_pairs": self.asymmetry_pairs[:20],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TONE_THRESHOLDS = (0.05, -0.05)   # compound > 0.05 → positive, < -0.05 → negative

NEUTRAL_THRESHOLD  = 0.05
ASYMMETRY_MIN_DIFF = 0.20   # minimum mean-score gap to flag as asymmetric


def _classify_tone(scores: List[float]) -> str:
    if not scores:
        return "neutral"
    mean = sum(scores) / len(scores)
    pos_count = sum(1 for s in scores if s > NEUTRAL_THRESHOLD)
    neg_count = sum(1 for s in scores if s < -NEUTRAL_THRESHOLD)

    if mean > NEUTRAL_THRESHOLD and pos_count > neg_count:
        return "positive"
    if mean < -NEUTRAL_THRESHOLD and neg_count > pos_count:
        return "negative"
    if pos_count > 0 and neg_count > 0:
        return "mixed"
    return "neutral"


def _context_window(sentences: List[str], idx: int, window: int = 1) -> str:
    """Return the sentence at idx plus up to `window` surrounding sentences."""
    start = max(0, idx - window)
    end = min(len(sentences), idx + window + 1)
    return " ".join(sentences[start:end])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_sentiment(
    preprocessed,
    sentences: Optional[List[str]] = None,
    target_labels: Optional[List[str]] = None,
    context_window: int = 1,
) -> SentimentResult:
    """
    Analyse sentiment toward named entities in a document.

    Args:
        preprocessed:   PreprocessedDocument from text_preprocessor.
        sentences:      Full sentence list (uses preprocessed.sentences if None).
        target_labels:  NER labels to analyse (default: PERSON, ORG, GPE).
        context_window: Number of surrounding sentences to include when
                        scoring entity sentiment (helps with attribution).

    Returns:
        SentimentResult with per-entity profiles and asymmetry detection.
    """
    if target_labels is None:
        target_labels = ["PERSON", "ORG", "GPE"]

    sents = sentences if sentences is not None else preprocessed.sentences
    scorer_name = "vader" if _get_vader() is not None else "lexicon"

    result = SentimentResult(source_id=preprocessed.source_id, scorer=scorer_name)

    # Score each entity mention in context
    entity_score_map: Dict[str, List[float]] = defaultdict(list)
    entity_label_map: Dict[str, str] = {}
    entity_sentiments: List[EntitySentiment] = []

    for ent in preprocessed.entities:
        if ent.label not in target_labels:
            continue
        context = _context_window(sents, ent.sentence_index, context_window)
        score = _score_sentence(context)

        key = ent.text.lower()
        entity_score_map[key].append(score)
        entity_label_map[key] = ent.label

        entity_sentiments.append(EntitySentiment(
            entity_text=ent.text,
            entity_label=ent.label,
            sentence=sents[ent.sentence_index] if ent.sentence_index < len(sents) else "",
            sentence_index=ent.sentence_index,
            score=score,
        ))

    result.entity_sentiments = entity_sentiments

    # Build entity profiles
    profiles: List[EntityProfile] = []
    for key, scores in entity_score_map.items():
        mean_score = sum(scores) / len(scores)
        profiles.append(EntityProfile(
            entity_text=key,
            entity_label=entity_label_map[key],
            mention_count=len(scores),
            scores=scores,
            mean_score=mean_score,
            tone=_classify_tone(scores),
        ))

    # Sort by mention count descending
    profiles.sort(key=lambda p: p.mention_count, reverse=True)
    result.entity_profiles = profiles

    # Detect asymmetric pairs
    result.asymmetry_pairs = _detect_asymmetry(profiles)

    return result


def _detect_asymmetry(profiles: List[EntityProfile]) -> List[dict]:
    """
    Find entity pairs with significantly different mean sentiment scores.

    Returns pairs sorted by absolute score difference (largest first).
    """
    pairs = []
    # Only compare entities with enough mentions to be meaningful
    significant = [p for p in profiles if p.mention_count >= 2]

    for i, a in enumerate(significant):
        for b in significant[i + 1:]:
            diff = abs(a.mean_score - b.mean_score)
            if diff >= ASYMMETRY_MIN_DIFF:
                pairs.append({
                    "entity_a": a.entity_text,
                    "entity_b": b.entity_text,
                    "score_a": round(a.mean_score, 4),
                    "score_b": round(b.mean_score, 4),
                    "difference": round(diff, 4),
                    "favoured": a.entity_text if a.mean_score > b.mean_score else b.entity_text,
                })

    pairs.sort(key=lambda x: x["difference"], reverse=True)
    return pairs
