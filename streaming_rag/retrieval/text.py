"""Shared text utilities. Deliberately dependency-free and deterministic:
the same string always produces the same tokens on any machine."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

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
    parts = [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
