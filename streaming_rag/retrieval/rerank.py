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
           coverage_weight: float = 0.6, density_weight: float = 0.15) -> list[tuple[Chunk, float]]:
    """candidates is (chunk, normalised_fused_score, indexed_text). Coverage is
    measured against the *indexed* text — the same string the retrievers scored —
    so a term that only appears in the document title still counts."""
    scored = []
    for chunk, fused_score, indexed_text in candidates:
        score = (fused_score
                 + coverage_weight * coverage(query, indexed_text)
                 + density_weight * factual_density(chunk.text))
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
