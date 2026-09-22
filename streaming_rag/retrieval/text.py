"""Shared text utilities. Tokenization/stopwords stay dependency-free and
deterministic: the same string always produces the same tokens on any
machine. Sentence splitting prefers spaCy's rule-based sentencizer (a blank
pipeline, no model download — still fully deterministic) when available,
because a plain regex on `.!?` mis-segments quoted rhetorical questions
mid-paragraph (e.g. `Tesla once wrote, "What can I say?" Tesla may have...`)
into a spurious tiny "sentence", which corrupts the synthesizer's
best-sentence extraction. Falls back to the regex splitter so the module
still works with zero dependencies installed.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_spacy_nlp = None
_spacy_unavailable = False


def _get_spacy_sentencizer():
    global _spacy_nlp, _spacy_unavailable
    if _spacy_unavailable:
        return None
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _spacy_nlp = nlp
        return nlp
    except Exception:
        _spacy_unavailable = True
        return None

STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with without
is are was were be been being am do does did doing have has had having i you he she it we they
me him her us them my your his its our their as so such not no nor can will would should could
about into over under again further once here there when where why how all any both each few more
most other some only own same too very s t just don now please need want
""".split())


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    nlp = _get_spacy_sentencizer()
    if nlp is not None:
        doc = nlp(stripped)
        parts = [s.text.strip() for s in doc.sents if s.text.strip()]
        if parts:
            return parts
    parts = [s.strip() for s in _SENTENCE_RE.split(stripped) if s.strip()]
    return parts or [stripped]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
