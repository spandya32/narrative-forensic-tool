"""
Narrative entropy calculator for the Narrative Forensics Tool.

Formula (CLAUDE.md §8):
    Entropy = - Σ (p_i * log(p_i))

Where p_i is the probability of each explanation cluster.

Algorithm:
  1. Embed all claim sentences using sentence-transformers (or TF-IDF)
  2. Cluster embeddings with KMeans (scikit-learn)
  3. Compute cluster probability distribution
  4. Apply Shannon entropy formula

Low entropy  → narrative consolidation (one dominant explanation)
High entropy → diverse explanations present

The optimal number of clusters is estimated automatically using the
elbow method (inertia curve) up to a configurable maximum.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntropyCluster:
    cluster_id: int
    size: int
    probability: float
    representative_sentence: str   # sentence closest to cluster centroid


@dataclass
class NarrativeEntropyResult:
    source_id: str
    entropy: float
    max_entropy: float             # log(n_clusters) — theoretical max
    normalised_entropy: float      # entropy / max_entropy  ∈ [0, 1]
    n_clusters: int
    clusters: List[EntropyCluster] = field(default_factory=list)
    sentences_used: int = 0
    embedder: str = ""

    @property
    def label(self) -> str:
        if self.normalised_entropy >= 0.75:
            return "high diversity"
        if self.normalised_entropy >= 0.45:
            return "moderate diversity"
        if self.normalised_entropy >= 0.20:
            return "low diversity"
        return "consolidated narrative"

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "entropy": round(self.entropy, 4),
            "max_entropy": round(self.max_entropy, 4),
            "normalised_entropy": round(self.normalised_entropy, 4),
            "n_clusters": self.n_clusters,
            "label": self.label,
            "sentences_used": self.sentences_used,
            "embedder": self.embedder,
            "clusters": [
                {
                    "id": c.cluster_id,
                    "size": c.size,
                    "probability": round(c.probability, 4),
                    "representative": c.representative_sentence[:200],
                }
                for c in self.clusters
            ],
        }

    def summary(self) -> str:
        return (
            f"Entropy Score : {self.entropy:.4f}\n"
            f"Normalised    : {self.normalised_entropy:.4f}  ({self.label})\n"
            f"Clusters      : {self.n_clusters}\n"
            f"Sentences     : {self.sentences_used}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(probabilities: List[float]) -> float:
    """Compute Shannon entropy for a probability distribution."""
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def _optimal_k(embeddings, max_k: int) -> int:
    """
    Estimate optimal number of clusters using the elbow method.
    Returns k in [2, max_k].
    """
    from sklearn.cluster import KMeans  # type: ignore
    import numpy as np  # type: ignore

    n = len(embeddings)
    if n <= 2:
        return 2

    max_k = min(max_k, n - 1)
    if max_k < 2:
        return 2

    inertias = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
        km.fit(embeddings)
        inertias.append(km.inertia_)

    # Find the elbow: largest second derivative
    if len(inertias) < 3:
        return ks[0]

    second_deriv = [
        inertias[i - 1] - 2 * inertias[i] + inertias[i + 1]
        for i in range(1, len(inertias) - 1)
    ]
    elbow_idx = second_deriv.index(max(second_deriv)) + 1  # +1 for offset
    return ks[elbow_idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MAX_CLUSTERS = 8
MAX_SENTENCES = 500   # cap to keep KMeans fast


def calculate_entropy(
    preprocessed,
    source_id: str = "",
    max_clusters: int = MAX_CLUSTERS,
) -> NarrativeEntropyResult:
    """
    Calculate narrative entropy from a PreprocessedDocument.

    Uses claim sentences as the unit of analysis since they carry
    the explanatory content of a text.

    Args:
        preprocessed:  PreprocessedDocument from text_preprocessor.
        source_id:     Label override.
        max_clusters:  Maximum number of explanation clusters to try.

    Returns:
        NarrativeEntropyResult.
    """
    sid = source_id or preprocessed.source_id

    # Use claim sentences; fall back to all sentences if too few claims
    claim_sents = [preprocessed.sentences[i] for i in preprocessed.claim_sentence_indices]
    if len(claim_sents) < 4:
        claim_sents = preprocessed.sentences

    claim_sents = [s for s in claim_sents if len(s.split()) >= 5][:MAX_SENTENCES]

    if len(claim_sents) < 4:
        return NarrativeEntropyResult(
            source_id=sid,
            entropy=0.0,
            max_entropy=0.0,
            normalised_entropy=0.0,
            n_clusters=0,
            sentences_used=len(claim_sents),
            embedder="none",
        )

    try:
        return _compute_entropy(claim_sents, sid, max_clusters)
    except Exception as exc:
        return NarrativeEntropyResult(
            source_id=sid,
            entropy=0.0,
            max_entropy=0.0,
            normalised_entropy=0.0,
            n_clusters=0,
            sentences_used=len(claim_sents),
            embedder=f"error: {exc}",
        )


def _compute_entropy(
    sentences: List[str],
    source_id: str,
    max_clusters: int,
) -> NarrativeEntropyResult:
    import numpy as np  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore

    # Embed
    from framing_detector import _embed_sentences, _embedder_name
    embeddings = _embed_sentences(sentences)
    embedder = _embedder_name()
    emb_array = np.array(embeddings, dtype=float)

    # Choose k
    k = _optimal_k(emb_array, max_clusters)

    # Cluster
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
    labels = km.fit_predict(emb_array)

    # Cluster sizes → probabilities
    counts = [0] * k
    for label in labels:
        counts[label] += 1
    n_total = len(sentences)
    probs = [c / n_total for c in counts]

    entropy = _shannon_entropy(probs)
    max_entropy = math.log(k) if k > 1 else 1.0
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0

    # Find representative sentence for each cluster (closest to centroid)
    clusters: List[EntropyCluster] = []
    for cid in range(k):
        cluster_indices = [i for i, l in enumerate(labels) if l == cid]
        centroid = km.cluster_centers_[cid]
        # Closest sentence to centroid
        best_idx = min(
            cluster_indices,
            key=lambda i: float(np.linalg.norm(emb_array[i] - centroid)),
        )
        clusters.append(EntropyCluster(
            cluster_id=cid,
            size=counts[cid],
            probability=probs[cid],
            representative_sentence=sentences[best_idx],
        ))

    clusters.sort(key=lambda c: c.size, reverse=True)

    return NarrativeEntropyResult(
        source_id=source_id,
        entropy=entropy,
        max_entropy=max_entropy,
        normalised_entropy=normalised,
        n_clusters=k,
        clusters=clusters,
        sentences_used=len(sentences),
        embedder=embedder,
    )
