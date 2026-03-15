"""
Framing change detection for the Narrative Forensics Tool.

Detects semantic shifts in how the same topic or entity is described,
both within a single document and between two documents.

Algorithm (CLAUDE.md §4):
  - Sentence embeddings via sentence-transformers (all-MiniLM-L6-v2)
  - Cosine similarity to measure semantic distance
  - Falls back to TF-IDF cosine similarity if sentence-transformers
    is not installed

Two modes:
  1. intra-doc  : detect framing variance for each named entity across
                  the sentences it appears in (one document)
  2. inter-doc  : detect framing shifts between matched sentence pairs
                  from two documents (uses diff_engine output)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FramingShift:
    entity: str                 # entity or topic being described
    text_a: str                 # original phrasing
    text_b: str                 # shifted phrasing
    similarity: float           # cosine similarity [0, 1]; lower = more shifted
    shift_score: float          # 1 - similarity; higher = more shifted
    source_a: str = ""
    source_b: str = ""


@dataclass
class FramingResult:
    source_id: str
    mode: str                   # "intra" | "inter"
    embedder: str               # "sentence-transformers" | "tfidf"
    shifts: List[FramingShift] = field(default_factory=list)
    mean_shift_score: float = 0.0
    intra_variance: float = 0.0  # average variance of entity sentence embeddings

    @property
    def shift_count(self) -> int:
        return len(self.shifts)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "mode": self.mode,
            "embedder": self.embedder,
            "mean_shift_score": round(self.mean_shift_score, 4),
            "intra_variance": round(self.intra_variance, 4),
            "shift_count": self.shift_count,
            "top_shifts": [
                {
                    "entity": s.entity,
                    "text_a": s.text_a[:200],
                    "text_b": s.text_b[:200],
                    "similarity": round(s.similarity, 4),
                    "shift_score": round(s.shift_score, 4),
                }
                for s in self.shifts[:20]
            ],
        }


# ---------------------------------------------------------------------------
# Embedder (lazy loader with TF-IDF fallback)
# ---------------------------------------------------------------------------

_st_model = None
_st_available: Optional[bool] = None


def _get_st_model():
    global _st_model, _st_available
    if _st_available is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        _st_available = True
    except Exception:
        _st_available = False
        _st_model = None
    return _st_model


def _embed_sentences(sentences: List[str]) -> "list[list[float]]":
    """Return embeddings as a list of float vectors."""
    model = _get_st_model()
    if model is not None:
        vecs = model.encode(sentences, show_progress_bar=False)
        return [v.tolist() for v in vecs]
    # TF-IDF fallback
    return _tfidf_embed(sentences)


def _tfidf_embed(sentences: List[str]) -> "list[list[float]]":
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    vec = TfidfVectorizer(max_features=500, sublinear_tf=True)
    try:
        mat = vec.fit_transform(sentences)
        return mat.toarray().tolist()
    except Exception:
        dim = 10
        return [[0.0] * dim for _ in sentences]


def _embedder_name() -> str:
    return "sentence-transformers" if _get_st_model() is not None else "tfidf"


def _cosine(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Minimum shift score to report a framing change
SHIFT_THRESHOLD = 0.20


def detect_intra_framing(preprocessed, source_id: str = "") -> FramingResult:
    """
    Detect framing variance within a single document.

    For each named entity that appears in ≥ 2 sentences, compute the
    average pairwise cosine distance between those sentences' embeddings.
    High variance → the entity is described in markedly different ways.

    Args:
        preprocessed: PreprocessedDocument from text_preprocessor.
        source_id:    Label for the document.

    Returns:
        FramingResult (mode="intra").
    """
    from collections import defaultdict

    sid = source_id or preprocessed.source_id
    result = FramingResult(source_id=sid, mode="intra", embedder=_embedder_name())

    sents = preprocessed.sentences
    # Group sentence indices by entity
    entity_sents: dict = defaultdict(list)
    for ent in preprocessed.entities:
        if ent.label in ("PERSON", "ORG", "GPE", "WORK_OF_ART"):
            key = ent.text.strip()
            idx = ent.sentence_index
            if idx < len(sents) and sents[idx].strip():
                entity_sents[key].append(idx)

    if not entity_sents:
        return result

    variances = []
    shifts: List[FramingShift] = []

    for entity, indices in entity_sents.items():
        unique_idx = list(dict.fromkeys(indices))  # preserve order, deduplicate
        if len(unique_idx) < 2:
            continue

        # Cap at 10 sentences per entity to keep things fast
        unique_idx = unique_idx[:10]
        entity_sentences = [sents[i] for i in unique_idx]

        embeddings = _embed_sentences(entity_sentences)

        # Compute pairwise similarities
        pair_similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = _cosine(embeddings[i], embeddings[j])
                pair_similarities.append((i, j, sim))

        if not pair_similarities:
            continue

        avg_sim = sum(s for _, _, s in pair_similarities) / len(pair_similarities)
        variance = 1.0 - avg_sim
        variances.append(variance)

        # Record the most divergent pair as the "shift" for this entity
        pair_similarities.sort(key=lambda x: x[2])  # lowest similarity first
        i, j, min_sim = pair_similarities[0]
        shift_score = 1.0 - min_sim

        if shift_score >= SHIFT_THRESHOLD:
            shifts.append(FramingShift(
                entity=entity,
                text_a=entity_sentences[i],
                text_b=entity_sentences[j],
                similarity=min_sim,
                shift_score=shift_score,
                source_a=sid,
                source_b=sid,
            ))

    shifts.sort(key=lambda s: s.shift_score, reverse=True)
    result.shifts = shifts
    result.mean_shift_score = sum(s.shift_score for s in shifts) / len(shifts) if shifts else 0.0
    result.intra_variance = sum(variances) / len(variances) if variances else 0.0
    return result


def detect_inter_framing(
    sentences_a: List[str],
    sentences_b: List[str],
    source_a: str = "doc_a",
    source_b: str = "doc_b",
    modified_pairs: Optional[List[Tuple[str, str]]] = None,
) -> FramingResult:
    """
    Detect framing shifts between two documents.

    Uses already-identified modified sentence pairs from diff_engine when
    available; otherwise embeds all sentences and finds closest cross-doc
    pairs with low similarity.

    Args:
        sentences_a:    Sentences from document A.
        sentences_b:    Sentences from document B.
        source_a:       Label for document A.
        source_b:       Label for document B.
        modified_pairs: Optional list of (sent_a, sent_b) already matched
                        by the diff engine.

    Returns:
        FramingResult (mode="inter").
    """
    result = FramingResult(
        source_id=f"{source_a} vs {source_b}",
        mode="inter",
        embedder=_embedder_name(),
    )

    if modified_pairs:
        pairs_to_score = modified_pairs[:100]
    else:
        # Sample up to 200 sentences from each doc and find cross matches
        sample_a = sentences_a[:200]
        sample_b = sentences_b[:200]
        if not sample_a or not sample_b:
            return result
        emb_a = _embed_sentences(sample_a)
        emb_b = _embed_sentences(sample_b)
        pairs_to_score = _find_closest_pairs(sample_a, sample_b, emb_a, emb_b, top_n=30)

    if not pairs_to_score:
        return result

    texts_a = [p[0] for p in pairs_to_score]
    texts_b = [p[1] for p in pairs_to_score]
    all_texts = texts_a + texts_b
    embeddings = _embed_sentences(all_texts)
    emb_a_list = embeddings[:len(texts_a)]
    emb_b_list = embeddings[len(texts_a):]

    shifts: List[FramingShift] = []
    for i, (ta, tb) in enumerate(zip(texts_a, texts_b)):
        sim = _cosine(emb_a_list[i], emb_b_list[i])
        shift_score = 1.0 - sim
        if shift_score >= SHIFT_THRESHOLD:
            shifts.append(FramingShift(
                entity="",
                text_a=ta,
                text_b=tb,
                similarity=sim,
                shift_score=shift_score,
                source_a=source_a,
                source_b=source_b,
            ))

    shifts.sort(key=lambda s: s.shift_score, reverse=True)
    result.shifts = shifts
    result.mean_shift_score = sum(s.shift_score for s in shifts) / len(shifts) if shifts else 0.0
    return result


def _find_closest_pairs(
    sents_a: List[str],
    sents_b: List[str],
    emb_a: list,
    emb_b: list,
    top_n: int = 30,
) -> List[Tuple[str, str]]:
    """For each sentence in A, find its nearest neighbour in B."""
    used_b = set()
    pairs = []
    for i, ea in enumerate(emb_a):
        best_sim = -1.0
        best_j = -1
        for j, eb in enumerate(emb_b):
            if j in used_b:
                continue
            sim = _cosine(ea, eb)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
            pairs.append((sents_a[i], sents_b[best_j]))
        if len(pairs) >= top_n:
            break
    return pairs
