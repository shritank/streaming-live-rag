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

_DISCOURSE_RE = re.compile(
    r"^(?:(?:actually|also|wait|oh|so|okay|ok|well|one\s+more\s+thing|it\s+turns\s+out(?:\s+that)?|"
    r"i\s+forgot\s+to\s+(?:say|mention)(?:\s+that)?)[\s,.:;—–-]*)+",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(
    r"^(?:what|who|whom|whose|which|when|where|why|how|is|are|was|were|do|does|did|can|could|"
    r"should|would|will|has|have|had)\b",
    re.IGNORECASE,
)


def strip_discourse_markers(text: str) -> str:
    return _DISCOURSE_RE.sub("", text.strip()).strip()


def is_self_contained(clause: str) -> bool:
    """A clause phrased as a full question with at least two content terms of
    its own ("What disease did Tesla catch?") is already searchable; bolting
    a sibling clause's topic onto it only dilutes the query. An elliptical
    clause ("the catering options") still needs the shared topic."""
    return bool(_QUESTION_START_RE.match(clause.strip())) and len(content_tokens(clause)) >= 2


def _clean(clause: str) -> str:
    text = clause.strip().strip(".,;:")
    text = _LEAD_FILLER_RE.sub("", text).strip()
    text = re.sub(r"^(?:the|a|an|my|our)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


_WH = frozenset("what who whom whose which when where why how".split())
_NO_SPLIT_BEFORE_WH = frozenset("""
in on at by for from of to with about upon into over under through during between among against
is was are were be been being am do does did know knows knew tell told ask asked say said see saw
decide decided explain explained wonder and or but the a an that this
""".split())
_PUNCT_RE = re.compile(r"[.?!,;:]")


def split_unpunctuated_questions(utterance: str, min_tokens: int = 3) -> str:
    """ASR text has no punctuation, so 'X dependent what second part of air
    Y' cannot be split on commas. Inserts ', ' before a question word that
    starts a new question: the running clause already has >= min_tokens
    content words and the preceding word is not one that normally governs a
    wh-word ("by what", "know what", "is what"). Only for text with no
    punctuation at all — punctuated input is left to the normal splitter."""
    if _PUNCT_RE.search(utterance):
        return utterance
    words = utterance.split()
    out: list[str] = []
    running, clause_is_question = 0, False
    for i, w in enumerate(words):
        lw = w.lower()
        # only split where the text so far is itself a question; "the Amazon
        # rainforest makes up what amount..." is ONE question with the
        # wh-word in place, not two
        if (lw in _WH and i > 0 and running >= min_tokens and clause_is_question
                and words[i - 1].lower() not in _NO_SPLIT_BEFORE_WH):
            out[-1] = out[-1] + ","
            running, clause_is_question = 0, False
        out.append(w)
        if lw in _WH or (running == 0 and _QUESTION_START_RE.match(lw)):
            clause_is_question = True
        if content_tokens(w):
            running += 1
        if lw == "and":
            running, clause_is_question = 0, False
    return " ".join(out)


_QUESTION_END_RE = re.compile(r"(?<=\?)[\s,]*(?:and\s+)?")


def _split_on_boundaries(utterance: str) -> list[str]:
    """Comma / 'and' boundaries separate spoken list items ("the refund
    policy, the catering options, and parking"), but inside a sentence that
    ends in '?' they are usually part of ONE question ("...Social Darwinism
    and theories of race?", "Along with diesel engines, what engines...?").
    There a boundary is kept only when the text on both sides opens a
    question ("What is X, and when was Y built?")."""
    pieces: list[str] = []
    for sentence in _QUESTION_END_RE.split(utterance):
        if not sentence.strip():
            continue
        parts = _SPLIT_RE.split(sentence)
        if not sentence.rstrip().endswith("?"):
            pieces += parts
            continue
        seps = [m.group(0) for m in _SPLIT_RE.finditer(sentence)]
        groups = [parts[0]]
        for sep, part in zip(seps, parts[1:]):
            if _QUESTION_START_RE.match(part.strip()) and _QUESTION_START_RE.match(groups[-1].strip()):
                groups.append(part)
            else:
                groups[-1] = groups[-1] + sep + part
        pieces += groups
    return pieces


def _split_raw(utterance: str) -> list[tuple[str, str]]:
    """(raw clause, cleaned clause) pairs. The min-content-token filter exists
    to drop junk fragments produced BY splitting ("and", "also" left over as
    their own clause) — it must not also reject a short but legitimate
    question: "When are the ashes now?" alone, or "What is CSNET" inside a
    compound request (one content token once "What is" is cleaned away). So
    the threshold applies only when there are several candidate clauses,
    and never to a clause phrased as a question."""
    pairs = [(raw.strip(), _clean(raw)) for raw in _split_on_boundaries(utterance)]
    pairs = [(raw, clean) for raw, clean in pairs if clean]
    if len(pairs) <= 1:
        return pairs
    return [(raw, clean) for raw, clean in pairs
            if len(content_tokens(clean)) >= _MIN_CLAUSE_TOKENS or _QUESTION_START_RE.match(raw)]


def split_clauses(utterance: str) -> list[str]:
    return [clean for _, clean in _split_raw(utterance)]


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
              merge_threshold: float = 0.6, strip_markers: bool = False,
              context_carry: str = "short", clause_sink: dict | None = None,
              normalize_numbers: bool = False, split_unpunctuated: bool = False) -> list[SubQuery]:
    """Returns only sub-queries whose content is not already covered.

    `already_covered` is the set of content tokens the controller has already
    issued queries for in this utterance, which is what stops a re-emission
    when a later chunk repeats an earlier clause (§6.2). If `clause_sink` is
    given, it receives query_id -> the clause text BEFORE any carried topic
    words were appended (what supersession must compare).
    """
    if strip_markers:
        utterance = strip_discourse_markers(utterance)
    if normalize_numbers:
        from .normalize import spoken_numbers_to_digits
        utterance = spoken_numbers_to_digits(utterance)
    if split_unpunctuated:
        utterance = split_unpunctuated_questions(utterance)
    pairs = _split_raw(utterance)
    if not pairs:
        return []

    context = shared_context([clean for _, clean in pairs])
    out: list[SubQuery] = []
    accepted_tokens: list[set[str]] = []

    for raw, clause in pairs:
        tokens = set(content_tokens(clause))
        if not tokens - already_covered:
            continue                                  # nothing new in this clause
        if any(jaccard(tokens, seen) >= merge_threshold for seen in accepted_tokens):
            continue                                  # near-duplicate of a sibling clause
        text = clause
        # judged on the RAW clause: cleaning strips "What is/are", which
        # would make a full question look elliptical
        needs_context = (len(tokens) < 4 if context_carry == "short"
                         else not is_self_contained(raw))
        if needs_context and context:
            carried = [t for t in context if t not in tokens]
            if carried:
                text = f"{clause} {' '.join(carried)}"
        query_id = f"{utterance_id}.q{next_index + len(out) + 1}"
        out.append(SubQuery(
            query_id=query_id,
            text=text,
            intent_label=_label(clause),
            trigger=trigger,
            utterance_id=utterance_id,
        ))
        if clause_sink is not None:
            clause_sink[query_id] = clause
        accepted_tokens.append(tokens)

    return out
