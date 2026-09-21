"""Intent-stability scoring.

The controller's hardest call is *when* a partial transcript carries enough
settled meaning to search on. Retrieving on every token thrashes the index
and pollutes the context window (guide §6 pitfall 1); waiting for the full
utterance throws away the whole latency advantage.

Stability is scored from three cheap, language-level signals, all computed
on the accumulated text so far:

  anchors    - does the fragment contain concrete, searchable content
               (a noun-ish content token, a number, a capitalised entity)?
  closure    - does it end mid-thought? A trailing preposition, conjunction
               or article means the next word will change the meaning.
  growth     - has the content stopped changing? If the last chunk added no
               new content tokens, the intent has settled.

No entity lists, no topic keywords: everything here is structural, so a
re-skinned hidden scenario (different cities, policies, numbers) scores the
same way.
"""
from __future__ import annotations

import re

from ..retrieval.text import content_tokens

# Function words that cannot end a complete thought.
_DANGLING = frozenset("""
in on at to for from by with about into over under of and or but the a an
my our your their its his her this that these those is are was were be been
i we you they he she it need want would like also plus including
""".split())

_NUMBER_RE = re.compile(r"\b\d+\b")
_CAPITALISED_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def has_strong_anchor(text: str) -> bool:
    """A concrete, searchable entity: a number, or a capitalised word that is
    not merely the first word of the sentence."""
    if _NUMBER_RE.search(text):
        return True
    words = text.split()
    return any(_CAPITALISED_RE.fullmatch(w.strip(".,;:")) for w in words[1:])


def has_weak_anchor(text: str) -> bool:
    """Enough common-noun substance to be worth a speculative search, even
    without a named entity."""
    return len(content_tokens(text)) >= 4


def ends_mid_thought(text: str) -> bool:
    stripped = text.strip().rstrip(",")
    if not stripped:
        return True
    if stripped[-1] in ".!?":
        return False
    last = re.findall(r"[A-Za-z']+", stripped)
    if not last:
        return True
    return last[-1].lower() in _DANGLING


def score(accumulated_text: str, previous_text: str, is_final: bool) -> float:
    """0..1 confidence that the intent is settled enough to retrieve on."""
    if is_final:
        return 1.0
    text = accumulated_text.strip()
    if not text:
        return 0.0

    value = 0.0
    if has_strong_anchor(text):
        value += 0.5
    elif has_weak_anchor(text):
        value += 0.25
    if not ends_mid_thought(text):
        value += 0.3

    new_tokens = set(content_tokens(text)) - set(content_tokens(previous_text))
    if previous_text and not new_tokens:
        value += 0.2          # nothing new arrived: the thought has settled
    elif len(content_tokens(text)) >= 5:
        value += 0.2          # enough substance to be worth a speculative search

    return min(1.0, value)
