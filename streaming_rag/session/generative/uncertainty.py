"""Explicit uncertainty.

The guide requires that a sub-intent with no usable evidence produces a visible
indicator rather than a confident guess. Three things feed it:

1. retrieval said ``low_confidence`` for a sub-query;
2. no surviving claim covers a sub-query;
3. the verifier dropped a claim as unsupported.

Over-hedging is also a failure (task context §7.3), so when every sub-intent is
covered this returns ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...contracts import Claim, SubQuery


@dataclass
class UncertaintyOutcome:
    text: str | None                 # the note to attach to the answer, if any
    unsupported_intents: list[str]   # query_ids with no supported claim
    needs_clarification: bool        # nothing at all was answerable


def _covered_query_ids(claims: list[Claim], claim_queries: dict[int, str | None]) -> set[str]:
    covered: set[str] = set()
    for index, claim in enumerate(claims):
        if claim.supported is False:
            continue
        query_id = claim_queries.get(index)
        if query_id:
            covered.add(query_id)
    return covered


def _covered_parents(sub_queries: list[SubQuery], covered: set[str]) -> set[str]:
    """Walk parent_query_id chains upward from every covered sub-query."""
    parents = {sq.query_id: sq.parent_query_id for sq in sub_queries}
    inherited: set[str] = set()
    for query_id in covered:
        parent = parents.get(query_id)
        while parent and parent not in inherited:
            inherited.add(parent)
            parent = parents.get(parent)
    return inherited


def build_uncertainty(sub_queries: list[SubQuery], kept_claims: list[Claim],
                      claim_queries: dict[int, str | None],
                      low_confidence_query_ids: set[str],
                      dropped_claims: list[Claim] | None = None) -> UncertaintyOutcome:
    """Work out what could not be verified, and phrase it."""
    if not sub_queries:
        return UncertaintyOutcome(None, [], needs_clarification=False)

    covered = _covered_query_ids(kept_claims, claim_queries)
    # A refinement delta answers the intent of the query it replaced, so
    # coverage propagates up the parent chain. Without this, every refinement
    # would report the original intent as unverified even after answering it.
    covered |= _covered_parents(sub_queries, covered)
    unsupported = [sq for sq in sub_queries
                   if sq.query_id not in covered or sq.query_id in low_confidence_query_ids]

    # A claim that covers nothing identifiable still counts as coverage when it
    # is the only claim; this avoids flagging every answer whose claims lack a
    # sub_query_id (some models omit it).
    if kept_claims and not covered and len(sub_queries) == 1:
        unsupported = [sq for sq in sub_queries if sq.query_id in low_confidence_query_ids]

    if not unsupported:
        return UncertaintyOutcome(None, [], needs_clarification=False)

    labels = [sq.intent_label or sq.text for sq in unsupported]
    needs_clarification = not kept_claims and len(unsupported) == len(sub_queries)

    if len(labels) == 1:
        subject = labels[0]
    else:
        subject = ", ".join(labels[:-1]) + f" and {labels[-1]}"

    was_were = "was" if len(labels) == 1 else "were"
    text = f"{subject.capitalize()} could not be verified from the retrieved corpus."
    if dropped_claims:
        text += (f" {len(dropped_claims)} statement"
                 f"{'s' if len(dropped_claims) != 1 else ''} {was_were} removed "
                 "because the cited sources did not support them.")
    return UncertaintyOutcome(text, [sq.query_id for sq in unsupported], needs_clarification)
