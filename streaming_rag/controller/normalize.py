"""Spoken-number normalisation for ASR transcripts.

ASR spells numbers out ("an eighteen thirty sonnet", "twelve million
viewers") while written corpora use digits ("1830", "12 million"), so a
lexical retriever cannot connect them. Conservative by design: a lone small
number word ("which one", "two of them") is left alone; runs of number words,
and single words worth 10 or more, are converted. Scale words million and
billion are kept as words because that is how text writes them.
"""
from __future__ import annotations

import re

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | {"hundred", "thousand", "oh"}
_KEEP_SCALE = {"million", "billion"}
_TOKEN_RE = re.compile(r"\S+")


def _below_hundred(words: list[str]) -> int | None:
    """'sixty three' / 'sixty' / 'three' / 'thirteen' -> int, else None."""
    if len(words) == 1:
        w = words[0]
        return _UNITS.get(w, _TENS.get(w))
    if len(words) == 2 and words[0] in _TENS and words[1] in _UNITS and _UNITS[words[1]] < 10:
        return _TENS[words[0]] + _UNITS[words[1]]
    return None


def _cardinal(words: list[str]) -> int | None:
    words = [w for w in words if w != "and"]
    if not words or "oh" in words:
        return None
    total, current = 0, 0
    i = 0
    while i < len(words):
        w = words[i]
        if w == "hundred":
            current = max(current, 1) * 100
        elif w == "thousand":
            total += max(current, 1) * 1000
            current = 0
        else:
            nxt = words[i + 1] if i + 1 < len(words) else None
            if w in _TENS and nxt in _UNITS and _UNITS[nxt] < 10:
                current += _TENS[w] + _UNITS[nxt]
                i += 1
            elif w in _UNITS or w in _TENS:
                current += _UNITS.get(w, _TENS.get(w))
            else:
                return None
        i += 1
    return total + current


def _year(words: list[str]) -> int | None:
    """'eighteen thirty' -> 1830, 'nineteen oh five' -> 1905,
    'nineteen hundred' -> 1900, 'nineteen sixty three' -> 1963."""
    for split in (1, 2):
        head, tail = words[:split], words[split:]
        century = _below_hundred(head)
        if century is None or not 10 <= century <= 20 or not tail:
            continue
        if tail == ["hundred"]:
            return century * 100
        if tail[0] == "oh" and len(tail) == 2 and tail[1] in _UNITS and _UNITS[tail[1]] < 10:
            return century * 100 + _UNITS[tail[1]]
        rest = _below_hundred(tail)
        if rest is not None and rest >= 10:
            return century * 100 + rest
    return None


def _convert(run: list[str]) -> str | None:
    words = [w for w in run]
    scale = None
    if words and words[-1] in _KEEP_SCALE:
        scale, words = words[-1], words[:-1]
    if not words:
        return None
    value = None
    if "thousand" not in words and "and" not in words:
        value = _year(words)
    if value is None:
        value = _cardinal(words)
    if value is None:
        return None
    if len(run) == 1 and value < 10:
        return None
    return f"{value} {scale}" if scale else str(value)


def spoken_numbers_to_digits(text: str) -> str:
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens):
            w = tokens[j].lower().strip(",.?!;:")
            is_num = w in _NUMBER_WORDS or w in _KEEP_SCALE or (w == "and" and j > i)
            if not is_num:
                break
            j += 1
        # never let a run end on a connector
        while j > i and tokens[j - 1].lower().strip(",.?!;:") in ("and", "oh"):
            j -= 1
        if j > i:
            run = [t.lower().strip(",.?!;:") for t in tokens[i:j]]
            converted = _convert(run)
            if converted is not None:
                out.append(converted + (tokens[j - 1][-1] if tokens[j - 1][-1] in ",.?!;:" else ""))
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)
