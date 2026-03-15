"""
Narrative mutation detector for the Narrative Forensics Tool — Phase 4.

Detects gradual wording changes in how a narrative phrase evolves
across documents or versions (CLAUDE.md §12).

Example:
  "temple ruins" → "temple-like remains" → "structures interpreted as temple"

Algorithm:
  1. For each shared narrative phrase cluster, embed all variant phrasings
  2. Compute cosine distance between each consecutive pair in chronological
     order (or document order when timestamps are unavailable)
  3. Flag chains where cumulative drift exceeds a threshold
     → these are mutation chains

Metric: cosine distance between phrase embeddings over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MutationStep:
    phrase_from: str
    phrase_to: str
    cosine_distance: float      # 0 = identical, 1 = maximally different
    source_from: str = ""
    source_to: str = ""


@dataclass
class MutationChain:
    seed_phrase: str            # original / earliest phrase
    steps: List[MutationStep] = field(default_factory=list)
    total_drift: float = 0.0    # cumulative cosine distance across all steps

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def mutation_label(self) -> str:
        if self.total_drift >= 0.60:
            return "high mutation"
        if self.total_drift >= 0.30:
            return "moderate mutation"
        return "low mutation"

    def to_dict(self) -> dict:
        return {
            "seed_phrase": self.seed_phrase,
            "total_drift": round(self.total_drift, 4),
            "mutation_label": self.mutation_label,
            "chain_length": self.length,
            "steps": [
                {
                    "from": s.phrase_from,
                    "to": s.phrase_to,
                    "distance": round(s.cosine_distance, 4),
                    "source_from": s.source_from,
                    "source_to": s.source_to,
                }
                for s in self.steps
            ],
        }


@dataclass
class MutationResult:
    source_ids: List[str]
    chains: List[MutationChain] = field(default_factory=list)
    embedder: str = ""

    @property
    def chain_count(self) -> int:
        return len(self.chains)

    @property
    def high_mutation_count(self) -> int:
        return sum(1 for c in self.chains if c.total_drift >= 0.60)

    def to_dict(self) -> dict:
        return {
            "source_ids": self.source_ids,
            "chain_count": self.chain_count,
            "high_mutation_count": self.high_mutation_count,
            "embedder": self.embedder,
            "chains": [c.to_dict() for c in self.chains[:20]],
        }

    def summary(self) -> str:
        return (
            f"Mutation chains detected : {self.chain_count}\n"
            f"High-mutation chains     : {self.high_mutation_count}\n"
            f"Embedder                 : {self.embedder}"
        )


# ---------------------------------------------------------------------------
# Cosine distance helper
# ---------------------------------------------------------------------------

def _cosine_distance(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    similarity = max(0.0, min(1.0, dot / (na * nb)))
    return 1.0 - similarity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DRIFT_THRESHOLD = 0.15   # minimum distance per step to count as a mutation


def detect_mutations(
    phrase_results_ordered: list,
    clusters=None,
) -> MutationResult:
    """
    Detect narrative mutation chains across ordered documents.

    Args:
        phrase_results_ordered: List of PhraseExtractionResult objects in
                                chronological / document order.
        clusters:               Optional PhraseClusters from phrase_clusterer
                                to use shared cluster centroids as seeds.

    Returns:
        MutationResult with identified mutation chains.
    """
    from framing_detector import _embed_sentences, _embedder_name

    embedder = _embedder_name()
    source_ids = [r.source_id for r in phrase_results_ordered]
    result = MutationResult(source_ids=source_ids, embedder=embedder)

    if len(phrase_results_ordered) < 2:
        return result

    # Collect top phrases per document as {phrase: source_id}
    doc_phrases: List[List[Tuple[str, str]]] = []
    for res in phrase_results_ordered:
        top = [(p.text, res.source_id) for p in res.top_phrases(40)]
        doc_phrases.append(top)

    # Find phrases that appear (possibly with variation) across documents
    # Strategy: embed all phrases from all docs, find nearest-neighbour
    # chains across consecutive documents
    all_texts = [p for doc in doc_phrases for p, _ in doc]
    all_sources = [s for doc in doc_phrases for _, s in doc]

    if not all_texts:
        return result

    embeddings = _embed_sentences(all_texts)

    # Build lookup: text → embedding
    emb_map = {text: embeddings[i] for i, text in enumerate(all_texts)}

    # For each phrase in doc[0], trace it through subsequent docs
    seen_seeds: set = set()
    chains: List[MutationChain] = []

    for phrase_a, src_a in doc_phrases[0]:
        if phrase_a in seen_seeds:
            continue
        seen_seeds.add(phrase_a)

        steps: List[MutationStep] = []
        current_phrase = phrase_a
        current_src = src_a
        total_drift = 0.0

        for doc_idx in range(1, len(doc_phrases)):
            candidates = doc_phrases[doc_idx]
            if not candidates:
                continue

            emb_current = emb_map.get(current_phrase)
            if emb_current is None:
                break

            # Find nearest phrase in next document
            best_dist = float("inf")
            best_phrase = None
            best_src = None

            for phrase_b, src_b in candidates:
                emb_b = emb_map.get(phrase_b)
                if emb_b is None:
                    continue
                dist = _cosine_distance(emb_current, emb_b)
                if dist < best_dist:
                    best_dist = dist
                    best_phrase = phrase_b
                    best_src = src_b

            if best_phrase is None:
                break

            # Only record if there is meaningful drift
            if best_dist >= DRIFT_THRESHOLD:
                steps.append(MutationStep(
                    phrase_from=current_phrase,
                    phrase_to=best_phrase,
                    cosine_distance=best_dist,
                    source_from=current_src,
                    source_to=best_src,
                ))
                total_drift += best_dist

            current_phrase = best_phrase
            current_src = best_src

        if steps:
            chains.append(MutationChain(
                seed_phrase=phrase_a,
                steps=steps,
                total_drift=total_drift,
            ))

    chains.sort(key=lambda c: c.total_drift, reverse=True)
    result.chains = chains
    return result
