"""
Text diff engine for the Narrative Forensics Tool.

Compares two text documents sentence-by-sentence using the Myers diff
algorithm (via Python's stdlib difflib) and classifies each change as:
  - added       : sentence present only in version B
  - removed     : sentence present only in version A
  - modified    : sentence substantially changed (similarity below threshold)
  - unchanged   : identical or near-identical sentences

Also computes a per-sentence similarity score using SequenceMatcher so
downstream modules (framing change detection, Phase 3) can build on the
raw diff output without re-computing it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SentenceChange:
    change_type: str        # "added" | "removed" | "modified" | "unchanged"
    text_a: str             # original sentence (empty if added)
    text_b: str             # new sentence (empty if removed)
    similarity: float       # 0.0 – 1.0; 1.0 = identical


@dataclass
class DiffResult:
    source_a: str
    source_b: str
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    modified: List[SentenceChange] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    all_changes: List[SentenceChange] = field(default_factory=list)

    # --- convenience counts ---
    @property
    def added_count(self) -> int:
        return len(self.added)

    @property
    def removed_count(self) -> int:
        return len(self.removed)

    @property
    def modified_count(self) -> int:
        return len(self.modified)

    @property
    def unchanged_count(self) -> int:
        return len(self.unchanged)

    @property
    def change_ratio(self) -> float:
        """Fraction of sentences in A that were changed or removed."""
        total_a = self.removed_count + self.modified_count + self.unchanged_count
        if total_a == 0:
            return 0.0
        return (self.removed_count + self.modified_count) / total_a

    def summary(self) -> str:
        return (
            f"Added   : {self.added_count}\n"
            f"Removed : {self.removed_count}\n"
            f"Modified: {self.modified_count}\n"
            f"Unchanged: {self.unchanged_count}\n"
            f"Change ratio (A): {self.change_ratio:.2%}"
        )

    def to_dict(self) -> dict:
        return {
            "source_a": self.source_a,
            "source_b": self.source_b,
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "modified_count": self.modified_count,
            "unchanged_count": self.unchanged_count,
            "change_ratio": round(self.change_ratio, 4),
            "added": self.added[:50],
            "removed": self.removed[:50],
            "modified": [
                {
                    "text_a": c.text_a[:200],
                    "text_b": c.text_b[:200],
                    "similarity": round(c.similarity, 4),
                }
                for c in self.modified[:50]
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SIMILARITY_CACHE: dict = {}


def _sentence_similarity(a: str, b: str) -> float:
    """Compute normalised similarity ratio between two strings."""
    key = (id(a), id(b))  # fast cache using object identity (same list elements)
    if key in _SIMILARITY_CACHE:
        return _SIMILARITY_CACHE[key]
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower(), autojunk=False).ratio()
    _SIMILARITY_CACHE[key] = ratio
    return ratio


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter used internally when no preprocessed doc is given."""
    import re
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip() and len(s.split()) >= 3]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MODIFY_THRESHOLD = 0.85   # similarity < this → "modified"; >= this → "unchanged"
MATCH_THRESHOLD  = 0.50   # minimum similarity to consider two sentences the same topic


def diff_texts(
    text_a: str,
    text_b: str,
    source_a: str = "version_a",
    source_b: str = "version_b",
    sentences_a: List[str] | None = None,
    sentences_b: List[str] | None = None,
    modify_threshold: float = MODIFY_THRESHOLD,
) -> DiffResult:
    """
    Compare two text documents and classify sentence-level changes.

    Args:
        text_a:           Plain text of document A (original / earlier).
        text_b:           Plain text of document B (new / later).
        source_a:         Label for document A.
        source_b:         Label for document B.
        sentences_a:      Pre-split sentences for A (skip splitting if provided).
        sentences_b:      Pre-split sentences for B.
        modify_threshold: Similarity ratio below which a matched pair is
                          labelled "modified" instead of "unchanged".

    Returns:
        DiffResult with categorised sentence changes.
    """
    sents_a = sentences_a if sentences_a is not None else _split_sentences(text_a)
    sents_b = sentences_b if sentences_b is not None else _split_sentences(text_b)

    result = DiffResult(source_a=source_a, source_b=source_b)

    # Use difflib.SequenceMatcher on the sentence lists.
    # We compare lowercase stripped versions for matching but keep originals.
    norm_a = [s.lower().strip() for s in sents_a]
    norm_b = [s.lower().strip() for s in sents_b]

    matcher = difflib.SequenceMatcher(None, norm_a, norm_b, autojunk=False)

    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            for sent in sents_a[a0:a1]:
                change = SentenceChange("unchanged", sent, sent, 1.0)
                result.unchanged.append(sent)
                result.all_changes.append(change)

        elif opcode == "insert":
            for sent in sents_b[b0:b1]:
                change = SentenceChange("added", "", sent, 0.0)
                result.added.append(sent)
                result.all_changes.append(change)

        elif opcode == "delete":
            for sent in sents_a[a0:a1]:
                change = SentenceChange("removed", sent, "", 0.0)
                result.removed.append(sent)
                result.all_changes.append(change)

        elif opcode == "replace":
            # Try to pair sentences within replaced blocks by similarity
            block_a = sents_a[a0:a1]
            block_b = sents_b[b0:b1]
            _align_and_classify(block_a, block_b, result, modify_threshold)

    return result


def _align_and_classify(
    block_a: List[str],
    block_b: List[str],
    result: DiffResult,
    modify_threshold: float,
) -> None:
    """
    Within a replaced block, align sentences by best similarity and classify
    each pair as modified/unchanged, or the remainder as added/removed.
    """
    used_b = set()

    for sent_a in block_a:
        best_sim = 0.0
        best_idx = -1
        for idx, sent_b in enumerate(block_b):
            if idx in used_b:
                continue
            sim = _sentence_similarity(sent_a, sent_b)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= MATCH_THRESHOLD:
            used_b.add(best_idx)
            sent_b = block_b[best_idx]
            if best_sim >= modify_threshold:
                change = SentenceChange("unchanged", sent_a, sent_b, best_sim)
                result.unchanged.append(sent_a)
            else:
                change = SentenceChange("modified", sent_a, sent_b, best_sim)
                result.modified.append(change)
            result.all_changes.append(change)
        else:
            # No matching sentence found in B → removed
            change = SentenceChange("removed", sent_a, "", 0.0)
            result.removed.append(sent_a)
            result.all_changes.append(change)

    # Remaining sentences in B that weren't matched → added
    for idx, sent_b in enumerate(block_b):
        if idx not in used_b:
            change = SentenceChange("added", "", sent_b, 0.0)
            result.added.append(sent_b)
            result.all_changes.append(change)


def diff_sentence_lists(
    sentences_a: List[str],
    sentences_b: List[str],
    source_a: str = "version_a",
    source_b: str = "version_b",
) -> DiffResult:
    """Convenience wrapper when sentences are already split."""
    return diff_texts("", "", source_a, source_b, sentences_a, sentences_b)
