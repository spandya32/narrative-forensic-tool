"""
Phrase embedding clusterer for the Narrative Forensics Tool — Phase 4.

Groups semantically similar narrative phrases into clusters using:
  - Sentence-transformers embeddings (or TF-IDF fallback)
  - Agglomerative clustering (scikit-learn) — better than KMeans for
    short text phrases where cluster count is not known in advance

Each cluster represents a distinct narrative theme. Cluster membership
across multiple documents reveals propagation paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PhraseCluster:
    cluster_id: int
    phrases: List[str]
    centroid_phrase: str        # phrase closest to cluster centre
    size: int
    source_ids: List[str]       # which documents contributed phrases here

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "centroid_phrase": self.centroid_phrase,
            "phrases": self.phrases[:15],
            "sources": list(set(self.source_ids)),
        }


@dataclass
class PhraseClusters:
    n_clusters: int
    clusters: List[PhraseCluster] = field(default_factory=list)
    embedder: str = ""
    cross_document: bool = False   # True when phrases come from multiple docs

    def to_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "embedder": self.embedder,
            "cross_document": self.cross_document,
            "clusters": [c.to_dict() for c in self.clusters],
        }

    def shared_clusters(self) -> List[PhraseCluster]:
        """Clusters whose phrases appear in more than one source document."""
        return [c for c in self.clusters if len(set(c.source_ids)) > 1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_phrases(
    phrase_results,
    distance_threshold: float = None,
) -> PhraseClusters:
    """
    Cluster narrative phrases from one or more PhraseExtractionResults.

    Args:
        phrase_results:    A single PhraseExtractionResult or a list of them.
        distance_threshold: Agglomerative clustering linkage threshold
                            (cosine distance; lower = tighter clusters).

    Returns:
        PhraseClusters with grouped PhraseCluster objects.
    """
    from phrase_extractor import PhraseExtractionResult

    from framing_detector import _get_st_model

    if isinstance(phrase_results, PhraseExtractionResult):
        phrase_results = [phrase_results]

    # Collect all phrases with their source
    all_phrases: List[str] = []
    all_sources: List[str] = []
    for res in phrase_results:
        for p in res.top_phrases(80):
            all_phrases.append(p.text)
            all_sources.append(p.source_id or res.source_id)

    if len(all_phrases) < 2:
        return PhraseClusters(n_clusters=0, embedder="none")

    cross_doc = len({s for s in all_sources}) > 1

    # Use a wider threshold for TF-IDF (cosine space is different from ST)
    if distance_threshold is None:
        distance_threshold = 0.30 if _get_st_model() is not None else 0.60

    try:
        return _cluster(all_phrases, all_sources, distance_threshold, cross_doc)
    except Exception as exc:
        return PhraseClusters(n_clusters=0, embedder=f"error: {exc}", cross_document=cross_doc)


def _cluster(
    phrases: List[str],
    sources: List[str],
    distance_threshold: float,
    cross_doc: bool,
) -> PhraseClusters:
    import numpy as np  # type: ignore
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    from framing_detector import _embed_sentences, _embedder_name

    embedder = _embedder_name()
    embeddings = _embed_sentences(phrases)
    emb_array = np.array(embeddings, dtype=float)

    # L2-normalise for cosine distance
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb_array / norms

    n = len(phrases)
    max_clusters = min(n - 1, 20)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(emb_norm)
    n_clusters = int(labels.max()) + 1

    # Build cluster objects
    clusters: List[PhraseCluster] = []
    for cid in range(n_clusters):
        indices = [i for i, l in enumerate(labels) if l == cid]
        cluster_phrases = [phrases[i] for i in indices]
        cluster_sources = [sources[i] for i in indices]

        # Centroid = mean of embeddings in this cluster
        cluster_emb = emb_norm[indices]
        centroid = cluster_emb.mean(axis=0)
        dists = np.linalg.norm(cluster_emb - centroid, axis=1)
        centroid_idx = indices[int(np.argmin(dists))]

        clusters.append(PhraseCluster(
            cluster_id=cid,
            phrases=cluster_phrases,
            centroid_phrase=phrases[centroid_idx],
            size=len(indices),
            source_ids=cluster_sources,
        ))

    clusters.sort(key=lambda c: c.size, reverse=True)

    return PhraseClusters(
        n_clusters=n_clusters,
        clusters=clusters,
        embedder=embedder,
        cross_document=cross_doc,
    )
