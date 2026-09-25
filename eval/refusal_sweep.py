"""python -m eval.refusal_sweep --scenarios DIR --corpus-dir DIR [--set k=v ...] --out FILE

Refusal / answer-selection analysis for single-question scenarios (e.g.
eval/heysquad.py output). Each scenario is run ONCE with the confidence gate
disabled (retrieval.ce_low_confidence_logit=-1e9). For every sub-query that
reached synthesis it records:

  passage     top reranker logit, and the margin to the second one
  sentence    best cross-encoder sentence score over the top-3 chunks
  claims      the lexical-selector claim and the cross-encoder-selector claim
  overlap     share of the question's content words found in the chosen
              sentence / in its whole section (SQuAD 2.0-style unanswerable
              questions are often perturbations whose key words are absent)

Every combination of selector and thresholds (and any classifier trained on
these features) can then be evaluated exactly, offline, with `outcome()`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from streaming_rag.retrieval.text import content_tokens

from .error_analysis import Analyzer
from .loader import discover_scenarios, load_ground_truth
from .quality import contains_answer
from .runner import run_scenario_detailed
from .stats import wilson


def _overlap(question: str, text: str) -> float:
    q = set(content_tokens(question))
    return len(q & set(content_tokens(text))) / len(q) if q else 0.0


def collect(scenarios: str, corpus_dir: str, overrides: list[str], time_scale: float) -> list[dict]:
    analyzer = Analyzer(corpus_dir, overrides + ["retrieval.ce_low_confidence_logit=-1000000000",
                                                 "synthesis.ce_claim_threshold=-1000000000"])
    lexical_cfg = analyzer.config
    items = []
    for path in discover_scenarios(scenarios):
        gt = load_ground_truth(path)["turns"]["u1"]
        calls: list = []
        asyncio.run(run_scenario_detailed(str(path), lexical_cfg, time_scale=time_scale, calls=calls))
        subs = []
        for call in calls:
            for sq in call["sub_queries"]:
                res = next((r for r in call["results"] if r.query_id == sq.query_id), None)
                if res is None or not res.evidence:
                    continue
                prev = analyzer.synth._config.synthesis.selector
                analyzer.synth._config.synthesis.selector = "lexical"
                lex, _ = analyzer.synth._claims_from([sq], [res])
                reader_claim, reader_margin = None, None
                if hasattr(analyzer.synth, "_ce_claim_original"):   # reader experiment (eval/model_zoo.py)
                    reader_claim = analyzer.synth._ce_claim(sq.text, res)
                    # the margins live in the module that defined the patched
                    # function — which is __main__ under `python -m eval.model_zoo`
                    margins = type(analyzer.synth)._ce_claim.__globals__.get("READER_MARGINS", {})
                    reader_margin = margins.get(reader_claim.text) if reader_claim else None
                    ce = analyzer.synth._ce_claim_original(sq.text, res)
                else:
                    ce = analyzer.synth._ce_claim(sq.text, res)
                analyzer.synth._config.synthesis.selector = prev
                scores = [e.score for e in res.evidence[:2]]
                section = analyzer.retriever.get_chunk_by_citation(res.evidence[0].chunk.citation)
                sentence_score = None
                if ce is not None:
                    from streaming_rag.retrieval.neural import get_cross_encoder
                    title = res.evidence[0].chunk.metadata.get("doc_title", "")
                    sentence_score = float(get_cross_encoder().score(sq.text, [f"{title}: {ce.text}"])[0])
                subs.append({
                    "query": sq.text,
                    "top": scores[0], "margin": scores[0] - scores[1] if len(scores) > 1 else 0.0,
                    "sentence": sentence_score,
                    "lex_claim": lex[0].text if lex else None, "lex_cit": lex[0].citations[0] if lex else None,
                    "ce_claim": ce.text if ce else None, "ce_cit": ce.citations[0] if ce else None,
                    "rd_claim": reader_claim.text if reader_claim else None,
                    "rd_cit": reader_claim.citations[0] if reader_claim else None,
                    "reader_margin": reader_margin,
                    "overlap_lex": _overlap(sq.text, lex[0].text) if lex else 0.0,
                    "overlap_ce": _overlap(sq.text, ce.text) if ce else 0.0,
                    "overlap_section": _overlap(sq.text, section.text) if section else 0.0,
                })
        row = analyzer.qrels.get(gt["gold_sub_intents"][0])
        items.append({"scenario": path.stem, "impossible": bool(gt.get("expect_uncertainty")),
                      "question": gt["gold_sub_intents"][0],
                      "answers": analyzer._usable(row) if row else [],
                      "gold": row["relevant"] if row else [], "subs": subs})
    return items


def outcome(item: dict, keep, selector: str = "lexical") -> tuple[bool, bool]:
    """(asserted anything, answered correctly); `keep(sub) -> bool` decides
    whether a sub-query's claim is asserted."""
    claim_key, cit_key = {"cross-encoder": ("ce_claim", "ce_cit"), "reader": ("rd_claim", "rd_cit")}.get(
        selector, ("lex_claim", "lex_cit"))
    kept = [s for s in item["subs"] if s[claim_key] and keep(s)]
    if not kept:
        return False, False
    text = " ".join(s[claim_key] for s in kept)
    if item["answers"]:
        return True, contains_answer(text, item["answers"])
    return True, any(s[cit_key] in item["gold"] for s in kept)


def summarize(items: list[dict], keep, selector: str = "lexical") -> dict:
    ans = [i for i in items if not i["impossible"]]
    una = [i for i in items if i["impossible"]]
    outcomes_a = [outcome(i, keep, selector) for i in ans]
    outcomes_u = [outcome(i, keep, selector) for i in una]
    correct = sum(o[1] for o in outcomes_a)
    wrong_answered = sum(1 for o in outcomes_a if o[0] and not o[1])
    refused_u = sum(1 for o in outcomes_u if not o[0])
    return {"answer_recall": (correct, len(ans)), "abstention": (refused_u, len(una)),
            "wrong_on_answerable": (wrong_answered, len(ans)),
            "balanced": (correct / max(len(ans), 1) + refused_u / max(len(una), 1)) / 2}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--time-scale", type=float, default=8.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    items = collect(args.scenarios, args.corpus_dir, args.overrides, args.time_scale)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"overrides": args.overrides, "items": items}, indent=1), encoding="utf-8")
    for sel in ("lexical", "cross-encoder"):
        for t in (-1e9, -5, -2, 0, 2, 4):
            r = summarize(items, lambda s, t=t: s["top"] >= t, sel)
            (a, na), (u, nu) = r["answer_recall"], r["abstention"]
            la, ha = wilson(a, na)
            print(f"{sel:13s} passage>={t:>6}: answer {a}/{na}={a / max(na, 1):.1%} [{la:.1%}-{ha:.1%}]  "
                  f"refuse-unanswerable {u}/{nu}={u / max(nu, 1):.1%}  balanced {r['balanced']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
