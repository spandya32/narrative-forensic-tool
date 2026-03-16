"""
Bias pattern detector for the Narrative Forensics Tool.

Detects systematic narrative bias through three statistical algorithms:

  1. Concession-Reversal Minimization
     Pattern: "although X committed atrocity, he was generally tolerant"
     The harm is acknowledged but its significance is immediately negated.

  2. Scope Minimizer Detection
     Pattern: "desecrated few temples", "only brahmins were affected"
     Quantifiers that shrink the scale or scope of harmful events.

  3. Asymmetric Scrutiny Scoring
     Measures analytical attention per entity/group:
       scrutiny = sentence_count × (hedge_density + citation_density + negative_sentiment)
     High variance across groups = structural bias in whose actions get interrogated.

These are purely structural/statistical signals. They identify *pattern* not *intent*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Concession-reversal: "although/while/despite/even though [clause], [positive claim]"
_CONCESSION = re.compile(
    r"\b(although|though|even though|while|whilst|despite|notwithstanding|"
    r"in spite of|regardless of)\b",
    re.IGNORECASE,
)

# Reversal/mitigation words following the concession
_REVERSAL = re.compile(
    r"\b(generally|overall|largely|mostly|on the whole|in general|by and large|"
    r"for the most part|broadly|essentially|fundamentally|predominantly|"
    r"tolerant|benevolent|enlightened|progressive|moderate|fair|just|"
    r"well-disposed|kind|gentle|magnanimous|generous|benign)\b",
    re.IGNORECASE,
)

# Harm/atrocity vocabulary that gets minimized
_HARM_WORDS = re.compile(
    r"\b(destroy(?:ed|ing)?|destroy|desecrat(?:ed|ing|ion)?|plunder(?:ed|ing)?|"
    r"massacre(?:d|ing)?|slaughter(?:ed|ing)?|kill(?:ed|ing)?|murder(?:ed|ing)?|"
    r"execut(?:ed|ing|ion)?|persecute(?:d|ing|ion)?|oppress(?:ed|ing|ion)?|"
    r"convert(?:ed|ing|ion)?\s+(?:by\s+)?force|forcible\s+conversion|"
    r"temple\s+(?:destruction|demolition|desecration)|loot(?:ed|ing)?|"
    r"pillage(?:d|ing)?|raid(?:ed|ing)?|atrocit(?:y|ies)|violence|brutality|"
    r"cruelty|oppression|subjugation|humiliation|persecution)\b",
    re.IGNORECASE,
)

# Scope minimizers — quantifiers that shrink harm
_SCOPE_MINIMIZERS = re.compile(
    r"\b(few|some|occasional(?:ly)?|minor|limited|isolated|sporadic|"
    r"rare(?:ly)?|select(?:ive(?:ly)?)?|certain|particular|specific|"
    r"only|merely|just|at most|no more than|a handful|a small number|"
    r"in some cases|in certain instances|to some extent|partially|"
    r"not all|not always|not necessarily|exceptions?\s+(?:exist|were))\b",
    re.IGNORECASE,
)

# Affected-group minimizers: "only X were affected / targeted"
_AFFECTED_MINIMIZER = re.compile(
    r"\b(only|merely|just|exclusively|solely)\b.{0,30}"
    r"\b(were|was|are|is)\b.{0,30}\b(affected|targeted|harmed|hurt|impacted|concerned)\b",
    re.IGNORECASE,
)

# Negative sentiment words for scrutiny scoring
_NEGATIVE_VOCAB = re.compile(
    r"\b(atrocit|massacre|murder|kill|destroy|brutal|cruel|oppress|"
    r"persecute|invade|plunder|loot|pillage|forced|coerce|subjugat|"
    r"tyran|despot|ruthless|barbaric|violence)\w*\b",
    re.IGNORECASE,
)

# Hedging density helper
_HEDGE_WORDS = re.compile(
    r"\b(may|might|could|would|perhaps|possibly|probably|"
    r"apparently|seemingly|allegedly|purportedly|claimed|suggested|"
    r"generally|usually|often|largely|mainly|mostly)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MinimizationInstance:
    sentence: str
    sentence_index: int
    concession_term: str       # "although", "despite", etc.
    harm_terms: List[str]      # harm words present
    reversal_terms: List[str]  # mitigation words present
    scope_terms: List[str]     # scope minimizers present
    severity: str              # "high" | "medium" | "low"

    def to_dict(self) -> dict:
        return {
            "sentence_index": self.sentence_index,
            "concession_term": self.concession_term,
            "harm_terms": self.harm_terms,
            "reversal_terms": self.reversal_terms,
            "scope_terms": self.scope_terms,
            "severity": self.severity,
            "sentence": self.sentence[:300],
        }


@dataclass
class ScrutinyProfile:
    entity: str
    mention_count: int
    sentences_about: int
    hedge_density: float       # hedging words per sentence about this entity
    negative_density: float    # negative vocab per sentence about this entity
    scrutiny_score: float      # composite: sentences × (hedge + negative)

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "mention_count": self.mention_count,
            "sentences_about": self.sentences_about,
            "hedge_density": round(self.hedge_density, 4),
            "negative_density": round(self.negative_density, 4),
            "scrutiny_score": round(self.scrutiny_score, 4),
        }


@dataclass
class AsymmetryPair:
    entity_high: str
    entity_low: str
    score_high: float
    score_low: float
    ratio: float          # high / low — how much more scrutiny one gets
    signal: str           # "over-scrutiny of A vs B" or "under-scrutiny of B vs A"

    def to_dict(self) -> dict:
        return {
            "entity_high_scrutiny": self.entity_high,
            "entity_low_scrutiny": self.entity_low,
            "score_high": round(self.score_high, 4),
            "score_low": round(self.score_low, 4),
            "ratio": round(self.ratio, 2),
            "signal": self.signal,
        }


@dataclass
class BiasDetectionResult:
    source_id: str

    # Algorithm 1: minimization
    minimization_instances: List[MinimizationInstance] = field(default_factory=list)
    minimization_count: int = 0
    high_severity_count: int = 0

    # Algorithm 2: scope minimizers
    scope_minimizer_sentences: List[dict] = field(default_factory=list)
    scope_minimizer_count: int = 0

    # Algorithm 3: asymmetric scrutiny
    scrutiny_profiles: List[ScrutinyProfile] = field(default_factory=list)
    asymmetry_pairs: List[AsymmetryPair] = field(default_factory=list)
    scrutiny_variance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "minimization_count": self.minimization_count,
            "high_severity_count": self.high_severity_count,
            "scope_minimizer_count": self.scope_minimizer_count,
            "scrutiny_variance": round(self.scrutiny_variance, 4),
            "asymmetry_pair_count": len(self.asymmetry_pairs),
            "minimization_instances": [m.to_dict() for m in self.minimization_instances[:10]],
            "scope_minimizer_sentences": self.scope_minimizer_sentences[:10],
            "scrutiny_profiles": [p.to_dict() for p in self.scrutiny_profiles[:15]],
            "asymmetry_pairs": [p.to_dict() for p in self.asymmetry_pairs[:10]],
        }


# ---------------------------------------------------------------------------
# Algorithm 1: Concession-reversal minimization
# ---------------------------------------------------------------------------

def _classify_severity(harm_count: int, reversal_count: int, scope_count: int) -> str:
    if harm_count >= 2 and reversal_count >= 1:
        return "high"
    if harm_count >= 1 and (reversal_count >= 1 or scope_count >= 1):
        return "medium"
    return "low"


def _detect_minimization(sentences: List[str]) -> List[MinimizationInstance]:
    results = []
    for idx, sent in enumerate(sentences):
        if not _CONCESSION.search(sent):
            continue

        harm_terms    = _HARM_WORDS.findall(sent)
        reversal_terms = _REVERSAL.findall(sent)
        scope_terms   = _SCOPE_MINIMIZERS.findall(sent)

        # Need at least a concession + either harm or reversal to be meaningful
        if not harm_terms and not reversal_terms:
            continue

        concession_match = _CONCESSION.search(sent)
        severity = _classify_severity(len(harm_terms), len(reversal_terms), len(scope_terms))

        results.append(MinimizationInstance(
            sentence=sent,
            sentence_index=idx,
            concession_term=concession_match.group(0).lower(),
            harm_terms=[t.lower() for t in harm_terms[:5]],
            reversal_terms=[t.lower() for t in reversal_terms[:5]],
            scope_terms=[t.lower() for t in scope_terms[:5]],
            severity=severity,
        ))

    # Sort: high severity first, then by presence of both harm + reversal
    results.sort(key=lambda x: (
        {"high": 0, "medium": 1, "low": 2}[x.severity],
        -(len(x.harm_terms) + len(x.reversal_terms)),
    ))
    return results


# ---------------------------------------------------------------------------
# Algorithm 2: Scope minimizer detection
# ---------------------------------------------------------------------------

def _detect_scope_minimizers(sentences: List[str]) -> List[dict]:
    results = []
    for idx, sent in enumerate(sentences):
        harm = _HARM_WORDS.findall(sent)
        scope = _SCOPE_MINIMIZERS.findall(sent)
        affected = _AFFECTED_MINIMIZER.search(sent)

        if not harm:
            continue
        if not scope and not affected:
            continue

        results.append({
            "sentence_index": idx,
            "harm_terms": [t.lower() for t in harm[:5]],
            "scope_terms": [t.lower() for t in scope[:5]],
            "affected_minimizer": bool(affected),
            "sentence": sent[:300],
        })

    return results


# ---------------------------------------------------------------------------
# Algorithm 3: Asymmetric scrutiny
# ---------------------------------------------------------------------------

def _compute_scrutiny(
    entities: List[dict],
    sentences: List[str],
) -> Tuple[List[ScrutinyProfile], List[AsymmetryPair], float]:
    """
    For each named entity, find sentences mentioning it and compute:
      hedge_density    = hedge words in those sentences / sentence count
      negative_density = negative vocab in those sentences / sentence count
      scrutiny_score   = sentence_count × (hedge_density + negative_density)
    """
    # Build entity → sentences mapping
    profiles: List[ScrutinyProfile] = []

    # Only entities with enough mentions to be meaningful
    significant = [e for e in entities if e.get("frequency", 0) >= 4][:30]

    for ent in significant:
        name = ent["text"].lower()
        ent_sents = [s for s in sentences if name in s.lower()]
        n = len(ent_sents)
        if n == 0:
            continue

        hedge_total = sum(len(_HEDGE_WORDS.findall(s)) for s in ent_sents)
        neg_total   = sum(len(_NEGATIVE_VOCAB.findall(s)) for s in ent_sents)
        word_total  = sum(len(s.split()) for s in ent_sents)

        hedge_density = hedge_total / max(word_total, 1) * 100
        neg_density   = neg_total   / max(word_total, 1) * 100
        scrutiny      = n * (hedge_density + neg_density)

        profiles.append(ScrutinyProfile(
            entity=ent["text"],
            mention_count=ent.get("frequency", n),
            sentences_about=n,
            hedge_density=hedge_density,
            negative_density=neg_density,
            scrutiny_score=scrutiny,
        ))

    profiles.sort(key=lambda p: p.scrutiny_score, reverse=True)

    # Compute variance
    scores = [p.scrutiny_score for p in profiles]
    if len(scores) >= 2:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    else:
        variance = 0.0

    # Build asymmetry pairs: top-scrutinized vs bottom-scrutinized
    pairs: List[AsymmetryPair] = []
    if len(profiles) >= 2:
        high_profiles = profiles[:5]
        low_profiles  = [p for p in profiles[-5:] if p.scrutiny_score > 0]
        for hi in high_profiles:
            for lo in low_profiles:
                if hi.entity == lo.entity:
                    continue
                if lo.scrutiny_score == 0:
                    continue
                ratio = hi.scrutiny_score / lo.scrutiny_score
                if ratio >= 3.0:   # only flag meaningful gaps
                    pairs.append(AsymmetryPair(
                        entity_high=hi.entity,
                        entity_low=lo.entity,
                        score_high=hi.scrutiny_score,
                        score_low=lo.scrutiny_score,
                        ratio=ratio,
                        signal=f"'{hi.entity}' receives {ratio:.1f}× more critical scrutiny than '{lo.entity}'",
                    ))

    pairs.sort(key=lambda p: p.ratio, reverse=True)
    return profiles, pairs[:10], variance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_bias(preprocessed, source_id: str = "") -> BiasDetectionResult:
    """
    Run all three bias detection algorithms on a PreprocessedDocument.

    Args:
        preprocessed: PreprocessedDocument from text_preprocessor.
        source_id:    Label override.

    Returns:
        BiasDetectionResult with all three algorithm outputs.
    """
    from text_preprocessor import extract_unique_entities

    sid   = source_id or preprocessed.source_id
    sents = preprocessed.sentences

    # Algorithm 1
    minimizations = _detect_minimization(sents)

    # Algorithm 2
    scope_hits = _detect_scope_minimizers(sents)

    # Algorithm 3
    entities = extract_unique_entities(preprocessed, labels=["PERSON", "ORG", "GPE"])
    profiles, pairs, variance = _compute_scrutiny(entities, sents)

    return BiasDetectionResult(
        source_id=sid,
        minimization_instances=minimizations[:20],
        minimization_count=len(minimizations),
        high_severity_count=sum(1 for m in minimizations if m.severity == "high"),
        scope_minimizer_sentences=scope_hits[:20],
        scope_minimizer_count=len(scope_hits),
        scrutiny_profiles=profiles,
        asymmetry_pairs=pairs,
        scrutiny_variance=variance,
    )
