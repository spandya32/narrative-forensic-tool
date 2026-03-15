"""
Text preprocessing module for the Narrative Forensics Tool.

Responsibilities:
  - Sentence tokenisation
  - Word tokenisation
  - Named Entity Recognition (spaCy)
  - Citation sentence detection (regex)
  - Claim sentence detection (heuristics)
  - Producing a PreprocessedDocument for downstream analysis modules

spaCy model: en_core_web_sm (small, CPU-friendly).
If spaCy is unavailable, the module degrades gracefully using basic
regex tokenisation; NER results will be empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    text: str
    label: str          # e.g. PERSON, ORG, GPE, DATE, WORK_OF_ART …
    sentence_index: int  # which sentence the entity appears in


@dataclass
class PreprocessedDocument:
    source_id: str              # file path or URL
    sentences: List[str] = field(default_factory=list)
    tokens: List[List[str]] = field(default_factory=list)   # tokens per sentence
    entities: List[Entity] = field(default_factory=list)
    citation_sentence_indices: List[int] = field(default_factory=list)
    claim_sentence_indices: List[int] = field(default_factory=list)
    ner_available: bool = False

    # Convenience properties -------------------------------------------------

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def citation_count(self) -> int:
        return len(self.citation_sentence_indices)

    @property
    def claim_count(self) -> int:
        return len(self.claim_sentence_indices)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "sentence_count": self.sentence_count,
            "citation_count": self.citation_count,
            "claim_count": self.claim_count,
            "ner_available": self.ner_available,
            "entities": [
                {"text": e.text, "label": e.label, "sentence_index": e.sentence_index}
                for e in self.entities
            ],
            "citation_sentence_indices": self.citation_sentence_indices,
            "claim_sentence_indices": self.claim_sentence_indices,
        }


# ---------------------------------------------------------------------------
# Front matter stripper
# ---------------------------------------------------------------------------

# Markers that signal the start of real content (end of front matter)
_CONTENT_MARKERS = re.compile(
    r"^(preface|introduction|chapter\s+1|chapter\s+one|foreword|"
    r"part\s+one|part\s+i\b|prologue|i\.\s+introduction)",
    re.IGNORECASE | re.MULTILINE,
)

# Lines that are pure front-matter noise
_FRONT_MATTER_LINE = re.compile(
    r"^("
    r"all rights reserved|copyright|isbn|printed in|published by|"
    r"first published|this edition|no part of this|reproduction|"
    r"retrieval system|without.*permission|typeset|cataloguing|"
    r"library of congress|british library|penguin|oxford university press|"
    r"table of contents|contents\s*$|acknowledgements?\s*$|"
    r"dedication\s*$|about the author|also by the same"
    r")",
    re.IGNORECASE,
)


def strip_front_matter(text: str) -> str:
    """
    Remove publisher boilerplate and front matter from extracted text.

    Strategy:
      1. Find the first real content marker (Preface, Introduction, Chapter 1…)
      2. Drop everything before it
      3. Remove individual lines that match front-matter patterns
    """
    # Find where real content starts
    match = _CONTENT_MARKERS.search(text)
    if match and match.start() > 100:          # only strip if there's actual front matter
        text = text[match.start():]

    # Filter remaining boilerplate lines
    lines = text.splitlines()
    clean_lines = [
        line for line in lines
        if not _FRONT_MATTER_LINE.match(line.strip())
    ]
    return "\n".join(clean_lines)


# ---------------------------------------------------------------------------
# Sentence tokenisation (regex fallback)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"
)


def _split_sentences_regex(text: str) -> List[str]:
    """Naive regex sentence splitter used when spaCy is unavailable."""
    raw = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in raw if s.strip()]


def _tokenise_sentence(sentence: str) -> List[str]:
    """Split a sentence into word-level tokens (lowercase, alpha only)."""
    return re.findall(r"\b[a-zA-Z]{2,}\b", sentence.lower())


# ---------------------------------------------------------------------------
# Citation detection
# ---------------------------------------------------------------------------

# Patterns that indicate a sentence contains a citation or reference
_CITATION_PATTERNS = [
    re.compile(r"\[\d+\]"),                                 # [1], [23]
    re.compile(r"\([\w\s]+,\s*\d{4}\)"),                   # (Smith, 2001)
    re.compile(r"according to", re.IGNORECASE),
    re.compile(r"cited in", re.IGNORECASE),
    re.compile(r"as noted by", re.IGNORECASE),
    re.compile(r"as stated by", re.IGNORECASE),
    re.compile(r"see also", re.IGNORECASE),
    re.compile(r"ibid", re.IGNORECASE),
    re.compile(r"op\.?\s*cit", re.IGNORECASE),
    re.compile(r"https?://\S+"),                            # bare URL
    re.compile(r"doi:\s*10\.\d{4}"),                        # DOI
]


def _is_citation_sentence(sentence: str) -> bool:
    return any(p.search(sentence) for p in _CITATION_PATTERNS)


# ---------------------------------------------------------------------------
# Claim detection (heuristic)
# ---------------------------------------------------------------------------

# Sentences that assert something (not just attribution or description)
_CLAIM_STARTERS = re.compile(
    r"\b(is|are|was|were|has|have|had|shows?|demonstrates?|proves?|"
    r"indicates?|suggests?|reveals?|established|confirmed|caused|"
    r"resulted in|led to|therefore|thus|hence|consequently|"
    r"it is (argued|believed|claimed|contended|maintained|proposed|"
    r"suggested|thought|widely held))\b",
    re.IGNORECASE,
)

# Sentences that are definitely NOT claims (pure questions / exclamations)
_NON_CLAIM = re.compile(r"^\s*(who|what|when|where|why|how)\b", re.IGNORECASE)


def _is_claim_sentence(sentence: str) -> bool:
    if _NON_CLAIM.match(sentence):
        return False
    # Require a minimum length (very short sentences are rarely claims)
    if len(sentence.split()) < 6:
        return False
    return bool(_CLAIM_STARTERS.search(sentence))


# ---------------------------------------------------------------------------
# spaCy loader (lazy, with graceful degradation)
# ---------------------------------------------------------------------------

_nlp = None          # cached spaCy pipeline
_spacy_available: Optional[bool] = None


def _get_nlp():
    global _nlp, _spacy_available
    if _spacy_available is not None:
        return _nlp

    try:
        import spacy  # type: ignore
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded yet – try a smaller built-in or skip NER
            _nlp = spacy.blank("en")
        _spacy_available = True
    except ImportError:
        _spacy_available = False
        _nlp = None

    return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(text: str, source_id: str = "") -> PreprocessedDocument:
    """
    Tokenise text, run NER, and classify sentences.

    Args:
        text:      Raw plain text (already stripped of markup).
        source_id: Identifier for the source document (path or URL).

    Returns:
        PreprocessedDocument ready for analysis modules.
    """
    doc = PreprocessedDocument(source_id=source_id)

    if not text or not text.strip():
        return doc

    text = strip_front_matter(text)

    nlp = _get_nlp()
    ner_available = _spacy_available and nlp is not None and nlp.has_pipe("ner")
    doc.ner_available = ner_available

    if ner_available:
        sentences, entities = _process_with_spacy(nlp, text)
        doc.entities = entities
    else:
        sentences = _split_sentences_regex(text)

    doc.sentences = sentences
    doc.tokens = [_tokenise_sentence(s) for s in sentences]

    for idx, sentence in enumerate(sentences):
        if _is_citation_sentence(sentence):
            doc.citation_sentence_indices.append(idx)
        if _is_claim_sentence(sentence):
            doc.claim_sentence_indices.append(idx)

    return doc


def _process_with_spacy(nlp, text: str) -> Tuple[List[str], List[Entity]]:
    """Use spaCy to split sentences and extract named entities."""
    MAX_CHARS = 500_000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    spacy_doc = nlp(text)

    # Use spaCy's own sentence boundaries — store (sent_index, sent_start_char)
    sent_list = [sent for sent in spacy_doc.sents if sent.text.strip()]
    sentences: List[str] = [s.text.strip() for s in sent_list]
    # Map each sentence's start char offset to its index
    sent_char_starts = [s.start_char for s in sent_list]

    entities: List[Entity] = []
    for ent in spacy_doc.ents:
        # Binary search for which sentence this entity belongs to
        sent_idx = 0
        for i, start in enumerate(sent_char_starts):
            if ent.start_char >= start:
                sent_idx = i
            else:
                break
        entities.append(Entity(text=ent.text.strip(), label=ent.label_, sentence_index=sent_idx))

    return sentences, entities


def extract_unique_entities(doc: PreprocessedDocument, labels: Optional[List[str]] = None) -> List[dict]:
    """
    Return deduplicated entity list, optionally filtered by label types.

    Args:
        doc:    A PreprocessedDocument.
        labels: Optional list of spaCy NER labels to include,
                e.g. ["PERSON", "ORG"]. None = include all.

    Returns:
        List of {text, label, frequency} dicts sorted by frequency desc.
    """
    from collections import Counter
    counts: Counter = Counter()
    label_map: dict = {}

    for ent in doc.entities:
        if labels is None or ent.label in labels:
            key = (ent.text.lower(), ent.label)
            counts[key] += 1
            label_map[key] = ent.label

    result = [
        {"text": text, "label": label, "frequency": freq}
        for (text, label), freq in counts.most_common()
    ]
    return result
