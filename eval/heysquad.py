"""python -m eval.heysquad --corpus-dir DIR --name NAME [--n-unanswerable N] [--seed S]

Builds a spoken-question benchmark for one corpus from HeySQuAD (Wu et al.,
2023; human-read SQuAD 2.0 questions with ASR transcriptions; CC BY 4.0),
validation split, text columns only (data/raw/heysquad_human_validation_text.jsonl,
see its .meta.json for the pinned dataset revision).

Two paired scenario sets are written, one utterance each:
  eval/scenarios_heysquad_<NAME>_clean/   the original typed question
  eval/scenarios_heysquad_<NAME>_asr/     the ASR transcription of a human
                                          reading that same question aloud
Answerable questions (SQuAD 1.1 ids present in the corpus qrels) are scored
with the gold-referenced metrics; SQuAD 2.0 unanswerable questions written
against this corpus's own passages must be declined ("expect_uncertainty").
These unanswerable questions are adversarial: they were written to look
answerable from the passage they sit next to.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from streaming_rag.controller.stream import chunk_utterance

from .quality import load_doc_titles, load_qrels_index

RAW = Path("data/raw/heysquad_human_validation_text.jsonl")
SQUAD = Path("data/raw/squad_validation.parquet")


def _has_early_opportunity(text: str, words_per_chunk: int = 6) -> bool:
    return len(text.split()) > words_per_chunk


def _scenario(sid: str, text: str, question: str, impossible: bool) -> dict:
    events = chunk_utterance(text, "u1", "s1", start_ms=0)
    last = events[-1]["timestamp_ms"] if events else 0
    return {
        "scenario_id": sid, "template": "heysquad",
        "events": [{"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}}]
                  + events
                  + [{"timestamp_ms": last + 2000, "event_type": "session_end", "payload": {"session_id": "s1"}}],
        "ground_truth": {"turns": {"u1": {
            "kind": "new_request", "retrieval_required": True,
            "eligible_for_early_retrieval": _has_early_opportunity(text),
            "gold_sub_intents": [question], "expect_uncertainty": impossible}}},
    }


def build(corpus_dir: str, name: str, n_unanswerable: int | None, seed: int) -> dict:
    import pandas as pd
    df = pd.read_parquet(SQUAD)
    id2q = dict(zip(df["id"], df["question"]))
    ctx2title = {}
    for c, t in zip(df["context"], df["title"]):
        ctx2title.setdefault(c[:200], t)
    qrels = load_qrels_index(corpus_dir)
    corpus_titles = {t.replace(" ", "_") for t in load_doc_titles(corpus_dir).values()}

    rows = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable, unanswerable = [], []
    for r in rows:
        impossible = str(r["is_impossible"]).lower() == "true"
        if not impossible and id2q.get(r["id"]) in qrels:
            answerable.append((r, id2q[r["id"]]))
        elif impossible and ctx2title.get(r["context_prefix"]) in corpus_titles:
            unanswerable.append((r, r["question"]))
    rng = random.Random(seed)
    if n_unanswerable is not None and len(unanswerable) > n_unanswerable:
        unanswerable = rng.sample(unanswerable, n_unanswerable)

    counts = {}
    for variant in ("clean", "asr"):
        out = Path(f"eval/scenarios_heysquad_{name}_{variant}")
        out.mkdir(parents=True, exist_ok=True)
        for items, impossible in ((answerable, False), (unanswerable, True)):
            for r, question in items:
                text = question if variant == "clean" else r["transcription"]
                sid = f"hey_{'u' if impossible else 'a'}_{r['id']}"
                (out / f"{sid}.json").write_text(json.dumps(_scenario(sid, text, question, impossible), indent=1),
                                                 encoding="utf-8")
        counts[variant] = len(answerable) + len(unanswerable)
    return {"answerable": len(answerable), "unanswerable": len(unanswerable), **counts}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-unanswerable", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(build(args.corpus_dir, args.name, args.n_unanswerable, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
