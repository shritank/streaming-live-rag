"""Multi-intent decomposition.

Turns a compound, unsegmented utterance into discrete search-ready
sub-queries. The two failure modes the guide calls out pull in opposite
directions, and both are guarded here:

  under-splitting  - one blended query retrieves mediocre evidence for
                     three different needs
  over-fragmenting - a simple question becomes several near-identical
                     queries that pollute the reranker (pitfall 5)

Strategy: split on coordination/list boundaries, then *reject* any candidate
that carries no new content of its own, and merge candidates that overlap
too heavily. Each surviving clause is rewritten into a standalone query by
carrying forward the utterance's shared context terms, because "the
cancellation policy" alone is not searchable — it needs the topic.
"""
from __future__ import annotations

import re

from ..contracts import SubQuery, Trigger
from ..retrieval.text import content_tokens, jaccard

_SPLIT_RE = re.compile(r",\s+and\s+|,\s+|\s+and\s+also\s+|\s+and\s+|\s+plus\s+|;\s*", re.IGNORECASE)
_LEAD_FILLER_RE = re.compile(
    r"^(?:i\s+(?:need|want|would\s+like|am\s+looking\s+for)|"
    r"can\s+you\s+(?:tell\s+me|find|give\s+me)|"
    r"please\s+(?:tell\s+me|find)|tell\s+me|what\s+(?:is|are)|"
    r"i'?m\s+looking\s+for)\s+(?:about\s+)?(?:to\s+)?",
    re.IGNORECASE,
)

# Words that make a clause a question about the topic rather than a new topic.
_MIN_CLAUSE_TOKENS = 2


def _clean(clause: str) -> str:
    text = clause.strip().strip(".,;:")
    text = _LEAD_FILLER_RE.sub("", text).strip()
    text = re.sub(r"^(?:the|a|an|my|our)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def split_clauses(utterance: str) -> list[str]:
    parts = [_clean(p) for p in _SPLIT_RE.split(utterance)]
    return [p for p in parts if len(content_tokens(p)) >= _MIN_CLAUSE_TOKENS]


def shared_context(clauses: list[str]) -> list[str]:
    """Content terms from the first clause act as the utterance's topic and are
    carried into later clauses that are too bare to search on their own."""
    if not clauses:
        return []
    return content_tokens(clauses[0])[:4]


def _label(clause: str) -> str:
    tokens = content_tokens(clause)
    return " ".join(tokens[:3]) if tokens else clause[:30]


def decompose(utterance: str, already_covered: set[str], utterance_id: str,
              trigger: Trigger, next_index: int,
              merge_threshold: float = 0.6) -> list[SubQuery]:
    """Returns only sub-queries whose content is not already covered.

    `already_covered` is the set of content tokens the controller has already
    issued queries for in this utterance, which is what stops a re-emission
    when a later chunk repeats an earlier clause (§6.2).
    """
    clauses = split_clauses(utterance)
    if not clauses:
        return []

    context = shared_context(clauses)
    out: list[SubQuery] = []
    accepted_tokens: list[set[str]] = []

    for clause in clauses:
        tokens = set(content_tokens(clause))
        if not tokens - already_covered:
            continue                                  # nothing new in this clause
        if any(jaccard(tokens, seen) >= merge_threshold for seen in accepted_tokens):
            continue                                  # near-duplicate of a sibling clause
        text = clause
        if len(tokens) < 4 and context:
            carried = [t for t in context if t not in tokens]
            if carried:
                text = f"{clause} {' '.join(carried)}"
        out.append(SubQuery(
            query_id=f"{utterance_id}.q{next_index + len(out) + 1}",
            text=text,
            intent_label=_label(clause),
            trigger=trigger,
            utterance_id=utterance_id,
        ))
        accepted_tokens.append(tokens)

    return out
