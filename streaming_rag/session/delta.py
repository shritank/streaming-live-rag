"""Answer delta engine (G5): refine, never restart.

When a late constraint arrives, the prior answer's claims are the starting
point. Only claims the new evidence actually touches are replaced; everything
else is carried forward verbatim, along with its citations. That is what
makes the refinement a *version* of the previous answer rather than a fresh
one, and it is what the G5 continuity check looks for:

  - version increments, parent_version points at the prior version
  - every prior citation still appears in the new answer
  - the retrieval for this version is scoped to the delta, not the whole topic
"""
from __future__ import annotations

from ..contracts import Claim
from ..retrieval.text import content_tokens, jaccard

SUPERSEDE_THRESHOLD = 0.45


def merge_claims(prior: list[Claim], delta: list[Claim]) -> tuple[list[Claim], list[str]]:
    """Returns (merged claims, citations carried over from prior).

    A delta claim supersedes a prior claim when they are about the same thing
    (high token overlap); otherwise it is appended as new information.
    """
    merged: list[Claim] = []
    superseded: set[int] = set()

    for d in delta:
        d_tokens = set(content_tokens(d.text))
        best_idx, best = None, 0.0
        for i, p in enumerate(prior):
            if i in superseded:
                continue
            overlap = jaccard(d_tokens, set(content_tokens(p.text)))
            if overlap > best:
                best, best_idx = overlap, i
        if best_idx is not None and best >= SUPERSEDE_THRESHOLD:
            superseded.add(best_idx)

    for i, p in enumerate(prior):
        if i not in superseded:
            merged.append(p)
    merged.extend(delta)

    carried = [c for p in prior for c in p.citations]
    return merged, carried


def union_citations(claims: list[Claim], carried: list[str]) -> list[str]:
    """Prior citations are kept even when their claim was superseded: the
    evidence still contributed to the answer's lineage (G5)."""
    seen: list[str] = []
    for citation in [c for claim in claims for c in claim.citations] + carried:
        if citation not in seen:
            seen.append(citation)
    return sorted(seen)
