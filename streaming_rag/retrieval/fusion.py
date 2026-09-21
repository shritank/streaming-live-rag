"""Reciprocal Rank Fusion.

RRF is used instead of score interpolation because BM25 scores and cosine
similarities are on incomparable scales; fusing on *rank* avoids having to
calibrate one against the other every time the corpus changes.
"""
from __future__ import annotations


def rrf(rank_lists: dict[str, list[int]], rrf_k: int = 60,
        weights: dict[str, float] | None = None) -> dict[int, float]:
    """rank_lists maps a source name ("dense"/"sparse") to an ordered list of
    document indices. Returns {doc_index: fused_score}, higher is better."""
    weights = weights or {}
    fused: dict[int, float] = {}
    for source, ranked in rank_lists.items():
        w = weights.get(source, 1.0)
        for rank, doc_index in enumerate(ranked):
            fused[doc_index] = fused.get(doc_index, 0.0) + w / (rrf_k + rank + 1)
    return fused
