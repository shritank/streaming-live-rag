"""python -m eval.asr_wer --scenarios DIR [--transcripts ASR.json]

Word error rate of a spoken-question scenario set against the question the
speaker read (the typed question is the reference transcript). Without
--transcripts, scores the text already in the scenarios (e.g. HeySQuAD's
provided ASR); with it, scores an ASR run's {id: {"text": ...}} output for
the same ids. Normalisation: lowercase, keep [a-z0-9'] tokens only, so a
transcript is not penalised for punctuation or casing it happens to carry.
Also reports the entity-token error rate: WER restricted to reference tokens
outside the stopword list (names, numbers, rare terms - what retrieval needs).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from streaming_rag.retrieval.text import STOPWORDS

from .loader import discover_scenarios


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, set[int]]:
    """Levenshtein distance and the set of reference positions that were not
    matched exactly (substituted or deleted)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    bad, i, j = set(), n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                bad.add(i - 1)
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            bad.add(i - 1)
            i -= 1
        else:
            j -= 1
    return d[n][m], bad


def score(pairs: list[tuple[str, str]]) -> dict:
    errs = words = ent_bad = ent = exact = 0
    for ref_text, hyp_text in pairs:
        ref, hyp = _tokens(ref_text), _tokens(hyp_text)
        dist, bad = edit_ops(ref, hyp)
        errs += dist
        words += len(ref)
        exact += dist == 0
        for k, w in enumerate(ref):
            if w not in STOPWORDS:
                ent += 1
                ent_bad += k in bad
    return {"n": len(pairs), "wer": errs / max(words, 1), "content_word_error_rate": ent_bad / max(ent, 1),
            "exact_transcripts": exact}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--transcripts")
    args = ap.parse_args(argv)
    asr = json.loads(Path(args.transcripts).read_text(encoding="utf-8"))["items"] if args.transcripts else None
    pairs = []
    for path in discover_scenarios(args.scenarios):
        sc = json.loads(Path(path).read_text(encoding="utf-8"))
        ref = sc["ground_truth"]["turns"]["u1"]["gold_sub_intents"][0]
        if asr is not None:
            rid = sc["scenario_id"].split("_", 2)[2]
            if rid not in asr:
                continue
            hyp = asr[rid]["text"]
        else:
            hyp = " ".join(e["payload"]["text"] for e in sc["events"] if e["event_type"] == "transcript_chunk")
        pairs.append((ref, hyp))
    print(json.dumps(score(pairs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
