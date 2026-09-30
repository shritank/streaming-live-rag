"""python -m eval.failure_taxonomy EA.json [EA.json ...] [--pair CLEAN.json ASR.json]
                                     [--modes STREAMING.json DEFERRED.json]

Maps eval.error_analysis records onto the failure taxonomy used for
bottleneck ranking, and computes what an oracle fix of each stage could
recover at most.

  A  retrieval failure            gold section not in the sub-query's evidence
  B  right document, wrong part   claim from the gold document but a non-gold
                                  section, or the answer sits in another chunk
  B2 wrong document selected      gold section WAS retrieved, the selector took
                                  a sentence from another document
  C  right chunk, wrong sentence
  D  extraction / label artefact  answer present in cited text but not matched,
                                  or a correct claim lost in the merge
  E  decomposition failure        no sub-query corresponds to the question
  F  cross-intent contamination   claim cited a section that is gold for a
                                  DIFFERENT intent of the same turn
  G  ASR corruption               (paired sets only) correct on the typed
                                  question, wrong on its ASR transcript
  H  refusal / abstention         a gate withheld the answer: learned refusal
                                  gate (H1), retrieval low confidence (H2),
                                  selector threshold (H3)
  I  citation failure             answer correct, gold section not cited
  J  streaming / supersession     (mode pair only) deferred right, streaming wrong
  K  other

"Recoverable" = an oracle fix of that stage alone would yield the correct
answer: the ungated claim was right (H), the answer sentence exists in the
claimed chunk (C), the answer is in the top-3 evidence (B, B2), the gold
section is in the fused top-20 or retrieved by the clean question (A).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

CLASS = {
    "first_stage": "A", "rerank_depth": "A", "query_formulation": "A",
    "context_selection": "B", "chunking": "B",
    "wrong_document": "B2",
    "sentence_selection": "C",
    "answer_not_in_text": "D", "merge_lost": "D",
    "decomposition": "E",
    "cross_intent_contamination": "F",
    "refusal_gate": "H1", "refusal": "H2", "relevance_gate": "H3",
}
NAMES = {
    "A": "retrieval failure", "B": "right doc, wrong section/chunk", "B2": "wrong document selected",
    "C": "right chunk, wrong sentence", "D": "extraction/label artefact", "E": "decomposition",
    "F": "cross-intent contamination", "H1": "learned refusal gate withheld", "H2": "retrieval low-confidence",
    "H3": "selector threshold", "K": "other",
}


def recoverable(rec: dict) -> bool:
    cause, d = rec["cause"], rec.get("detail", {})
    if cause in ("refusal_gate", "refusal", "relevance_gate"):
        return bool(d.get("ungated_correct"))
    if cause == "sentence_selection":
        return True
    if cause in ("context_selection", "chunking", "wrong_document", "cross_intent_contamination"):
        return bool(rec.get("answer_in_top3"))
    if cause in ("rerank_depth", "query_formulation"):
        return True
    if cause == "decomposition":
        return True          # upper bound: a perfect sub-query still has to retrieve and select
    return False


def taxonomy(records: list[dict]) -> dict:
    n = len(records)
    misses = [r for r in records if not r["correct"]]
    classes = Counter(CLASS.get(r["cause"], "K") for r in misses)
    rec_counts = Counter(CLASS.get(r["cause"], "K") for r in misses if recoverable(r))
    cite_fail = sum(1 for r in records if r["correct"] and not r.get("cited_gold", True))
    rows = []
    for c, k in classes.most_common():
        rows.append({"class": c, "name": NAMES.get(c, c), "misses": k, "pct_of_misses": k / max(len(misses), 1),
                     "recoverable": rec_counts.get(c, 0), "recoverable_pts": rec_counts.get(c, 0) / n})
    rows.sort(key=lambda r: -r["recoverable"])
    return {"n": n, "correct": n - len(misses), "misses": len(misses), "classes": rows,
            "I_answer_right_citation_wrong": cite_fail}


def oracle(records: list[dict]) -> dict:
    """Stage-wise upper bounds, each applied ALONE on top of the current system."""
    n = len(records)
    cur = sum(r["correct"] for r in records)
    by_stage = Counter()
    for r in records:
        if r["correct"]:
            continue
        cls = CLASS.get(r["cause"], "K")
        stage = {"E": "decomposition", "A": "retrieval", "H1": "refusal", "H2": "refusal", "H3": "refusal",
                 "B": "evidence_selection", "B2": "evidence_selection", "F": "evidence_selection",
                 "C": "sentence_selection"}.get(cls)
        if stage and recoverable(r):
            by_stage[stage] += 1
    return {"current": cur / n, "n": n,
            **{f"oracle_{k}": (cur + v) / n for k, v in by_stage.items()},
            "gain_pts": {k: v / n for k, v in by_stage.most_common()}}


def paired_asr(clean: list[dict], asr: list[dict]) -> dict:
    """G: questions answered from the typed text but not from its ASR transcript."""
    c = {r["question"]: r["correct"] for r in clean}
    a = {r["question"]: r for r in asr}
    both = [q for q in c if q in a]
    lost = [q for q in both if c[q] and not a[q]["correct"]]
    gained = [q for q in both if not c[q] and a[q]["correct"]]
    lost_by_stage = Counter(CLASS.get(a[q]["cause"], "K") for q in lost)
    return {"n_paired": len(both), "clean_correct": sum(c[q] for q in both),
            "asr_correct": sum(a[q]["correct"] for q in both),
            "G_lost_to_asr": len(lost), "gained_under_asr": len(gained),
            "oracle_asr_pts": (len(lost) - len(gained)) / max(len(both), 1),
            "where_asr_losses_surface": dict(lost_by_stage.most_common())}


def streaming_vs_deferred(stream_path: str, deferred_path: str) -> dict:
    s = json.load(open(stream_path, encoding="utf-8"))["items"]["answer"]
    d = json.load(open(deferred_path, encoding="utf-8"))["items"]["answer"]
    keys = set(s) & set(d)
    return {"n": len(keys), "J_deferred_right_streaming_wrong": sum(1 for k in keys if d[k] and not s[k]),
            "streaming_right_deferred_wrong": sum(1 for k in keys if s[k] and not d[k])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pair", nargs=2, metavar=("CLEAN", "ASR"))
    ap.add_argument("--modes", nargs=2, metavar=("STREAMING", "DEFERRED"))
    args = ap.parse_args(argv)
    out = {}
    for f in args.files:
        recs = json.load(open(f, encoding="utf-8"))["records"]
        out[f] = {"taxonomy": taxonomy(recs), "oracle": oracle(recs)}
    if args.pair:
        out["paired_asr"] = paired_asr(*(json.load(open(p, encoding="utf-8"))["records"] for p in args.pair))
    if args.modes:
        out["streaming_vs_deferred"] = streaming_vs_deferred(*args.modes)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
