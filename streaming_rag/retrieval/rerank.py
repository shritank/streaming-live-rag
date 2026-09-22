"""Reranking and deduplication of fused candidates.

Two jobs, both aimed at the guide's "merging multi-source evidence without
diluting context or introducing contradictions":

  1. rerank  - promote chunks that actually answer the query: term coverage
               plus factual density (numbers, units, modal obligations),
               which is what policy answers are usually made of.
  2. dedup   - drop near-duplicate chunks so three paraphrases of the same
               sentence don't crowd out a second, distinct fact.
"""
from __future__ import annotations

import re

from ..contracts import Chunk
from .text import content_tokens, jaccard

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_MODALS = frozenset({"must", "may", "requires", "required", "shall", "cannot", "prohibited", "eligible"})

# "What does AC stand for?" / "What is AC?" / "Define AC" — a definitional
# question about a short acronym-like target. Generalisable, not corpus-
# specific: encyclopedic text conventionally introduces an acronym once,
# right after its expansion, as "full term (ACRONYM)". A term-overlap
# reranker alone under-ranks the defining passage whenever the acronym
# itself is used far more often elsewhere in the document (e.g. "AC" recurs
# throughout a Tesla biography, but the definition appears exactly once).
_DEFINITIONAL_QUERY_RE = re.compile(
    r"\b(?:what\s+does\s+(?P<term1>[A-Za-z]{2,6})\s+stand\s+for"
    r"|what\s+is\s+(?:an?\s+)?(?P<term2>[A-Za-z]{2,6})\b"
    r"|define\s+(?P<term3>[A-Za-z]{2,6})\b)",
    re.IGNORECASE,
)


def definitional_target(query: str) -> str | None:
    """The acronym/term a "what does X stand for"-shaped query is asking
    about, or None if the query isn't that shape."""
    m = _DEFINITIONAL_QUERY_RE.search(query)
    if not m:
        return None
    term = m.group("term1") or m.group("term2") or m.group("term3")
    return term


def definitional_match(term: str, text: str) -> bool:
    """True if `text` contains the conventional "expansion (ACRONYM)" or
    "ACRONYM (expansion)" definitional pattern for `term`."""
    escaped = re.escape(term)
    pattern = re.compile(
        rf"(?:\([^()]*\b{escaped}\b[^()]*\))"      # "full term (AC)" — acronym inside parens
        rf"|(?:\b{escaped}\b\s*\([^()]*\))",        # "AC (full term)" — acronym before parens
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def factual_density(text: str) -> float:
    tokens = content_tokens(text)
    if not tokens:
        return 0.0
    numbers = len(_NUMBER_RE.findall(text))
    modals = sum(1 for t in tokens if t in _MODALS)
    return min(1.0, (numbers + modals) / max(len(tokens) / 8, 1))


def coverage(query: str, text: str) -> float:
    q = set(content_tokens(query))
    if not q:
        return 0.0
    return len(q & set(content_tokens(text))) / len(q)


def rerank(query: str, candidates: list[tuple[Chunk, float, str]],
           coverage_weight: float = 0.6, density_weight: float = 0.15,
           definitional_weight: float = 0.8) -> list[tuple[Chunk, float]]:
    """candidates is (chunk, normalised_fused_score, indexed_text). Coverage is
    measured against the *indexed* text — the same string the retrievers scored —
    so a term that only appears in the document title still counts."""
    term = definitional_target(query)
    scored = []
    for chunk, fused_score, indexed_text in candidates:
        score = (fused_score
                 + coverage_weight * coverage(query, indexed_text)
                 + density_weight * factual_density(chunk.text))
        if term and definitional_match(term, indexed_text):
            score += definitional_weight
        scored.append((chunk, score))
    scored.sort(key=lambda t: (-t[1], t[0].chunk_id))
    return scored


def dedup(candidates: list[tuple[Chunk, float]], threshold: float = 0.8) -> list[tuple[Chunk, float]]:
    """Keeps the highest-scoring representative of each near-duplicate cluster.
    Input must already be sorted best-first."""
    kept: list[tuple[Chunk, float]] = []
    kept_tokens: list[set[str]] = []
    for chunk, score in candidates:
        tokens = set(content_tokens(chunk.text))
        if any(jaccard(tokens, seen) >= threshold for seen in kept_tokens):
            continue
        kept.append((chunk, score))
        kept_tokens.append(tokens)
    return kept
