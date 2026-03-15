"""
Information Hierarchy & Salience Analyzer for the Narrative Forensics Tool.

Detects "lede burial" — the manipulation technique where key facts are
technically present and cited, but architecturally hidden by:

  1. Positional Burial       — key cited facts appear deep in the document
  2. Contextual Pressure     — claim surrounded by hedging/attribution/counter-claims
  3. Framing-Before-Fact     — interpretive frames precede evidence (primes reader)
  4. Lede Inversion          — framing sentences appear before evidentiary sentences
  5. Attribution Laundering  — facts wrapped in nested attribution chains

None of these techniques require false statements. The manipulation is purely
architectural — true, cited facts made effectively invisible to casual readers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Attribution / distancing phrases
_ATTRIBUTION = re.compile(
    r"\b("
    r"according to|stated by|claimed by|argued by|suggested by|"
    r"as noted by|as claimed|as stated|as argued|as suggested|"
    r"in the view of|in the opinion of|from the perspective of|"
    r"it is (?:claimed|alleged|suggested|argued|believed|thought|said)|"
    r"some (?:scholars|historians|experts|researchers) (?:claim|argue|suggest|believe|maintain)|"
    r"(?:historians|scholars|experts|critics) (?:have |who )?(?:argued|claimed|suggested|maintained)|"
    r"(?:it has been|has been) (?:argued|claimed|suggested|stated|asserted)"
    r")\b",
    re.IGNORECASE,
)

# Counter-claim / dispute markers
_COUNTER = re.compile(
    r"\b("
    r"however|but|on the other hand|in contrast|nevertheless|"
    r"disputed|contested|challenged|questioned|rejected|denied|"
    r"others? (?:argue|claim|suggest|contend)|"
    r"counter(?:ed)?|refuted|contradicted|disagree|not all"
    r")\b",
    re.IGNORECASE,
)

# Hedge words (same domain as hedging_detector but inline here for windowed use)
_HEDGE_INLINE = re.compile(
    r"\b("
    r"may|might|could|would|possibly|probably|perhaps|"
    r"appears?|seems?|suggests?|indicates?|implies?|"
    r"allegedly|purportedly|apparently|ostensibly|"
    r"believed|thought|considered|interpreted as|"
    r"generally|usually|often|largely|mainly|mostly"
    r")\b",
    re.IGNORECASE,
)

# Deep attribution nesting — "X said that Y found that Z claimed that"
_NESTED_ATTRIBUTION = re.compile(
    r"\b(?:said|stated|found|claimed|argued|noted|reported|concluded|"
    r"determined|established|discovered|revealed)\s+that\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BuriedFact:
    sentence: str
    sentence_index: int
    position: float            # 0.0 = document start, 1.0 = document end
    salience_score: float      # 1 - position; higher = more prominent
    contextual_pressure: float # hedge + attribution density in ±5 window
    attribution_depth: int     # count of attribution phrases in sentence
    is_cited: bool             # sentence also contains a citation marker

    def to_dict(self) -> dict:
        return {
            "sentence_index": self.sentence_index,
            "position_pct": round(self.position * 100, 1),
            "salience_score": round(self.salience_score, 4),
            "contextual_pressure": round(self.contextual_pressure, 4),
            "attribution_depth": self.attribution_depth,
            "is_cited": self.is_cited,
            "sentence": self.sentence[:200],
        }


@dataclass
class SalienceResult:
    source_id: str
    total_sentences: int

    # Algorithm 4: Lede Inversion
    mean_evidentiary_position: float   # where cited claims appear on average
    mean_framing_position: float       # where attribution/framing appears on average
    lede_inversion_score: float        # framing_pos - evidentiary_pos (negative = inverted)
    lede_inversion_label: str

    # Algorithm 1 + 2: Buried facts
    buried_facts: List[BuriedFact] = field(default_factory=list)

    # Algorithm 3: Framing-before-fact sequences
    framing_before_fact_count: int = 0
    framing_before_fact_examples: List[dict] = field(default_factory=list)

    # Algorithm 5: Attribution laundering
    max_laundering_depth: int = 0
    laundering_examples: List[str] = field(default_factory=list)

    # Summary
    mean_contextual_pressure: float = 0.0
    high_pressure_claim_count: int = 0  # claims with pressure > 0.4

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "total_sentences": self.total_sentences,
            "lede_inversion_score": round(self.lede_inversion_score, 4),
            "lede_inversion_label": self.lede_inversion_label,
            "mean_evidentiary_position": round(self.mean_evidentiary_position, 4),
            "mean_framing_position": round(self.mean_framing_position, 4),
            "buried_fact_count": len(self.buried_facts),
            "framing_before_fact_count": self.framing_before_fact_count,
            "max_laundering_depth": self.max_laundering_depth,
            "mean_contextual_pressure": round(self.mean_contextual_pressure, 4),
            "high_pressure_claim_count": self.high_pressure_claim_count,
            "buried_facts": [f.to_dict() for f in self.buried_facts[:10]],
            "laundering_examples": self.laundering_examples[:5],
            "framing_before_fact_examples": self.framing_before_fact_examples[:5],
        }

    def summary(self) -> str:
        return (
            f"Lede inversion      : {self.lede_inversion_score:+.4f}  ({self.lede_inversion_label})\n"
            f"Mean evid. position : {self.mean_evidentiary_position:.2%} through document\n"
            f"Mean framing pos.   : {self.mean_framing_position:.2%} through document\n"
            f"Buried key facts    : {len(self.buried_facts)}\n"
            f"High-pressure claims: {self.high_pressure_claim_count}\n"
            f"Max laundering depth: {self.max_laundering_depth}\n"
            f"Framing-before-fact : {self.framing_before_fact_count} sequences"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pressure_in_window(sentences: List[str], center: int, window: int = 5) -> float:
    """
    Compute contextual pressure in a ±window around sentence[center].

    Pressure = mean density of hedge + attribution + counter-claim words
    in the window sentences (excluding center itself).
    """
    start = max(0, center - window)
    end   = min(len(sentences), center + window + 1)
    context = sentences[start:center] + sentences[center + 1:end]
    if not context:
        return 0.0

    total_tokens = 0
    pressure_tokens = 0
    for sent in context:
        tokens = sent.split()
        total_tokens += len(tokens)
        pressure_tokens += len(_HEDGE_INLINE.findall(sent))
        pressure_tokens += len(_ATTRIBUTION.findall(sent))
        pressure_tokens += len(_COUNTER.findall(sent))

    return pressure_tokens / max(total_tokens, 1)


def _attribution_depth(sentence: str) -> int:
    """Count attribution phrases in a single sentence (laundering depth)."""
    return len(_ATTRIBUTION.findall(sentence))


def _nesting_depth(sentence: str) -> int:
    """Count 'said that / found that / claimed that' chains (nesting depth)."""
    return len(_NESTED_ATTRIBUTION.findall(sentence))


def _is_citation_sentence(sentence: str) -> bool:
    """Quick check if sentence contains a citation marker."""
    return bool(
        re.search(r"\[\d+\]|\(\w[\w\s]+,\s*\d{4}\)|https?://|doi:", sentence, re.IGNORECASE)
    )


def _lede_inversion_label(score: float) -> str:
    """
    Score = mean_framing_position - mean_evidentiary_position

    Positive  → framing appears AFTER evidence (normal, neutral)
    Negative  → framing appears BEFORE evidence (inverted lede, manipulation signal)
    Near zero → mixed / uniform distribution
    """
    if score < -0.15:
        return "strong inversion (framing precedes evidence)"
    if score < -0.05:
        return "mild inversion"
    if score > 0.10:
        return "normal (evidence before framing)"
    return "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_salience(preprocessed, source_id: str = "") -> SalienceResult:
    """
    Detect information hierarchy manipulation in a PreprocessedDocument.

    Runs all 5 algorithms:
      1. Positional salience scoring
      2. Contextual pressure indexing
      3. Framing-before-fact sequence detection
      4. Lede inversion measurement
      5. Attribution laundering depth

    Args:
        preprocessed: PreprocessedDocument from text_preprocessor.
        source_id:    Label override.

    Returns:
        SalienceResult with all metrics.
    """
    sid   = source_id or preprocessed.source_id
    sents = preprocessed.sentences
    total = len(sents)

    if total < 10:
        return SalienceResult(
            source_id=sid, total_sentences=total,
            mean_evidentiary_position=0.0, mean_framing_position=0.0,
            lede_inversion_score=0.0, lede_inversion_label="insufficient data",
        )

    claim_indices  = set(preprocessed.claim_sentence_indices)
    citation_indices = {
        i for i, s in enumerate(sents) if _is_citation_sentence(s)
    }
    # Framing sentences: attribution-heavy, not claims
    framing_indices = {
        i for i, s in enumerate(sents)
        if _ATTRIBUTION.search(s)
    }

    # --- Algorithm 4: Lede Inversion ---
    evid_positions = [i / total for i in claim_indices & citation_indices]
    frame_positions = [i / total for i in framing_indices - claim_indices]

    mean_evid  = sum(evid_positions)  / max(len(evid_positions), 1)
    mean_frame = sum(frame_positions) / max(len(frame_positions), 1)
    # Positive score = framing after evidence (normal)
    # Negative score = framing before evidence (inverted)
    inversion  = mean_frame - mean_evid

    # --- Algorithms 1 + 2: Buried facts ---
    buried: List[BuriedFact] = []
    pressures: List[float] = []

    for idx in sorted(claim_indices):
        position = idx / total
        pressure = _pressure_in_window(sents, idx)
        pressures.append(pressure)
        depth    = _attribution_depth(sents[idx])
        cited    = idx in citation_indices

        # A fact is "buried" if it's in the bottom half AND under high pressure
        # OR it's deeply attributed (laundering ≥ 2) regardless of position
        if (position > 0.50 and pressure > 0.03) or depth >= 2 or (cited and position > 0.65):
            buried.append(BuriedFact(
                sentence=sents[idx],
                sentence_index=idx,
                position=position,
                salience_score=1.0 - position,
                contextual_pressure=pressure,
                attribution_depth=depth,
                is_cited=cited,
            ))

    # Sort by a combined burial score: late position + high pressure + deep attribution
    buried.sort(
        key=lambda f: (1 - f.salience_score) + f.contextual_pressure + f.attribution_depth * 0.1,
        reverse=True,
    )

    high_pressure = sum(1 for p in pressures if p > 0.04)
    mean_pressure = sum(pressures) / max(len(pressures), 1)

    # --- Algorithm 3: Framing-before-fact ---
    fbf_count    = 0
    fbf_examples = []

    for idx in sorted(claim_indices):
        # Look at 3 sentences before this claim
        preceding = sents[max(0, idx - 3): idx]
        framing_before = [s for s in preceding if _ATTRIBUTION.search(s)]
        if framing_before:
            fbf_count += 1
            if len(fbf_examples) < 5:
                fbf_examples.append({
                    "framing": framing_before[-1][:150],
                    "claim": sents[idx][:150],
                    "claim_position_pct": round((idx / total) * 100, 1),
                })

    # --- Algorithm 5: Attribution laundering ---
    laundering_depths = []
    laundering_examples = []

    for i, sent in enumerate(sents):
        depth = _nesting_depth(sent)
        attr  = _attribution_depth(sent)
        combined = depth + attr
        if combined >= 3:
            laundering_depths.append(combined)
            if len(laundering_examples) < 5:
                laundering_examples.append(sent[:200])

    max_laundering = max(laundering_depths) if laundering_depths else 0

    return SalienceResult(
        source_id=sid,
        total_sentences=total,
        mean_evidentiary_position=mean_evid,
        mean_framing_position=mean_frame,
        lede_inversion_score=inversion,
        lede_inversion_label=_lede_inversion_label(inversion),
        buried_facts=buried[:20],
        framing_before_fact_count=fbf_count,
        framing_before_fact_examples=fbf_examples,
        max_laundering_depth=max_laundering,
        laundering_examples=laundering_examples,
        mean_contextual_pressure=mean_pressure,
        high_pressure_claim_count=high_pressure,
    )
