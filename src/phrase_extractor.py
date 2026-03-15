"""
Narrative phrase extractor for the Narrative Forensics Tool — Phase 4.

Steps (CLAUDE.md §11):
  1. Extract candidate phrases using n-grams (bigrams + trigrams)
  2. Filter with TF-IDF to keep only high-signal phrases
  3. Return ranked NarrativePhrase objects for downstream clustering
     and propagation tracking

The unit of analysis is a "narrative phrase": a short recurring
sequence of words that carries substantive meaning about events,
actors, or interpretations — not stopword noise.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NarrativePhrase:
    text: str
    frequency: int
    tfidf_score: float
    ngram_size: int             # 2 = bigram, 3 = trigram, etc.
    source_id: str = ""
    sentence_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "frequency": self.frequency,
            "tfidf_score": round(self.tfidf_score, 4),
            "ngram_size": self.ngram_size,
            "source_id": self.source_id,
        }


@dataclass
class PhraseExtractionResult:
    source_id: str
    phrases: List[NarrativePhrase] = field(default_factory=list)
    total_sentences: int = 0
    vocab_size: int = 0

    @property
    def phrase_count(self) -> int:
        return len(self.phrases)

    def top_phrases(self, n: int = 20) -> List[NarrativePhrase]:
        return sorted(self.phrases, key=lambda p: p.tfidf_score, reverse=True)[:n]

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "phrase_count": self.phrase_count,
            "total_sentences": self.total_sentences,
            "top_phrases": [p.to_dict() for p in self.top_phrases(30)],
        }


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "that", "this", "these", "those", "it", "its", "they", "them",
    "their", "which", "who", "whom", "what", "when", "where", "how",
    "why", "than", "then", "also", "just", "only", "even", "such",
    "into", "onto", "upon", "about", "above", "below", "between",
    "among", "through", "during", "before", "after", "while",
    "although", "because", "since", "unless", "until", "whether",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "nevertheless", "nonetheless", "meanwhile", "thereafter",
}

_TOKEN_RE = re.compile(r"\b[a-zA-Z]{2,}\b")


def _tokenise(sentence: str) -> List[str]:
    tokens = _TOKEN_RE.findall(sentence.lower())
    return [t for t in tokens if t not in _STOP_WORDS]


def _extract_ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# TF-IDF scorer
# ---------------------------------------------------------------------------

def _compute_tfidf(
    phrase_doc_counts: Counter,      # phrase → number of docs it appears in
    phrase_corpus_counts: Counter,   # phrase → total occurrences across all docs
    n_docs: int,
    doc_length: int,
) -> dict:
    """
    Compute TF-IDF scores for phrases within a single document context.

    TF  = phrase frequency in this doc / doc length
    IDF = log(n_docs / (1 + doc_freq))   [add 1 to avoid div-by-zero]
    """
    import math
    scores = {}
    for phrase, count in phrase_corpus_counts.items():
        tf = count / max(doc_length, 1)
        df = phrase_doc_counts.get(phrase, 1)
        idf = math.log(n_docs / (1 + df))
        scores[phrase] = tf * idf
    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MIN_PHRASE_FREQ = 2          # discard phrases seen only once
MIN_TFIDF      = 0.0         # keep all non-negative scores (filter by rank)
TOP_N_PHRASES  = 200         # max phrases returned per document


def extract_phrases(
    sentences: List[str],
    source_id: str = "",
    ngram_range: Tuple[int, int] = (2, 3),
    top_n: int = TOP_N_PHRASES,
) -> PhraseExtractionResult:
    """
    Extract high-signal narrative phrases from a list of sentences.

    Args:
        sentences:   Plain-text sentence list.
        source_id:   Label for the source document.
        ngram_range: (min_n, max_n) n-gram sizes to consider.
        top_n:       Maximum number of phrases to return.

    Returns:
        PhraseExtractionResult with TF-IDF-ranked NarrativePhrase objects.
    """
    result = PhraseExtractionResult(
        source_id=source_id,
        total_sentences=len(sentences),
    )
    if not sentences:
        return result

    min_n, max_n = ngram_range

    # Build per-sentence n-gram lists and global counts
    phrase_freq: Counter = Counter()
    phrase_sent_index: dict = {}    # phrase → list of sentence indices

    all_tokenised = [_tokenise(s) for s in sentences]

    for idx, tokens in enumerate(all_tokenised):
        if len(tokens) < min_n:
            continue
        seen_in_sent = set()
        for n in range(min_n, max_n + 1):
            for phrase in _extract_ngrams(tokens, n):
                if len(phrase.split()) < min_n:
                    continue
                phrase_freq[phrase] += 1
                if phrase not in seen_in_sent:
                    seen_in_sent.add(phrase)
                    phrase_sent_index.setdefault(phrase, []).append(idx)

    # Keep only phrases that appear at least MIN_PHRASE_FREQ times
    frequent = {p: c for p, c in phrase_freq.items() if c >= MIN_PHRASE_FREQ}

    result.vocab_size = len(frequent)

    if not frequent:
        return result

    # TF-IDF: treat each sentence as a "document" for IDF
    # doc_freq[phrase] = number of sentences containing the phrase
    doc_freq: Counter = Counter()
    for phrase in frequent:
        doc_freq[phrase] = len(phrase_sent_index.get(phrase, []))

    tfidf_scores = _compute_tfidf(
        phrase_doc_counts=doc_freq,
        phrase_corpus_counts=Counter(frequent),
        n_docs=len(sentences),
        doc_length=sum(len(t) for t in all_tokenised),
    )

    # Build NarrativePhrase objects, sorted by TF-IDF descending
    phrases = []
    for phrase, score in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        n = len(phrase.split())
        phrases.append(NarrativePhrase(
            text=phrase,
            frequency=frequent[phrase],
            tfidf_score=score,
            ngram_size=n,
            source_id=source_id,
            sentence_indices=phrase_sent_index.get(phrase, [])[:20],
        ))

    result.phrases = phrases
    return result


def compare_phrases(
    result_a: PhraseExtractionResult,
    result_b: PhraseExtractionResult,
) -> dict:
    """
    Compare phrase sets from two documents.

    Returns which high-signal phrases appear in both, only in A, only in B.
    """
    top_a = {p.text for p in result_a.top_phrases(50)}
    top_b = {p.text for p in result_b.top_phrases(50)}

    shared    = sorted(top_a & top_b)
    only_in_a = sorted(top_a - top_b)
    only_in_b = sorted(top_b - top_a)

    overlap_ratio = len(shared) / max(len(top_a | top_b), 1)

    return {
        "shared_phrases": shared[:30],
        "only_in_a": only_in_a[:30],
        "only_in_b": only_in_b[:30],
        "overlap_ratio": round(overlap_ratio, 4),
    }
