"""python -m eval.analyze_sweeps --dev A.json [B.json] --check C.json [D.json] [--ood E.json]

Offline analysis of eval.refusal_sweep outputs (with the reader recorded):

 1. selector comparison: lexical vs cross-encoder vs reader answer recall,
    paired, per file (no confidence gate)
 2. refusal: for each signal (passage logit, sentence score, reader margin,
    and a logistic combination fitted on DEV only), the threshold is chosen on
    DEV by a rule fixed in advance — maximise the balanced score
    (answer recall on answerable + refusal rate on unanswerable) / 2 — and
    then applied unchanged to the CHECK files.

Also reports the confidently-wrong rate: asserted-but-wrong answers over all
questions (answerable questions answered wrongly + unanswerable ones answered).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .refusal_sweep import outcome
from .stats import paired_bootstrap, wilson

SELECTORS = ("lexical", "cross-encoder", "reader")


def load(paths):
    items = []
    for p in paths:
        for it in json.load(open(p, encoding="utf-8"))["items"]:
            it["_file"] = p
            items.append(it)
    return items


def best_sub(item, selector):
    key = {"lexical": "lex_claim", "cross-encoder": "ce_claim", "reader": "rd_claim"}[selector]
    subs = [s for s in item["subs"] if s[key]]
    return max(subs, key=lambda s: s["top"]) if subs else None


FEATURES = ["top", "sentence", "reader_margin", "overlap_ce", "overlap_section", "margin",
            "nli_contra_rd", "nli_entail_rd"]


def features(item):
    s = best_sub(item, "reader") or best_sub(item, "cross-encoder")
    if s is None:
        return None
    defaults = {"sentence": -10.0, "reader_margin": -10.0, "nli_contra_rd": 0.5, "nli_entail_rd": 0.5}
    return [s.get(f) if s.get(f) is not None else defaults.get(f, 0.0) for f in FEATURES]


def evaluate(items, selector, assert_fn):
    """assert_fn(item) -> bool decides whether the system answers at all."""
    ans = [i for i in items if not i["impossible"]]
    una = [i for i in items if i["impossible"]]
    keep_all = lambda s: True
    correct = sum(1 for i in ans if assert_fn(i) and outcome(i, keep_all, selector)[1])
    wrong_ans = sum(1 for i in ans if assert_fn(i) and outcome(i, keep_all, selector)[0]
                    and not outcome(i, keep_all, selector)[1])
    answered_una = sum(1 for i in una if assert_fn(i) and outcome(i, keep_all, selector)[0])
    n = len(items)
    return {"answer_recall": (correct, len(ans)), "refusal": (len(una) - answered_una, len(una)),
            "confidently_wrong": (wrong_ans + answered_una, n),
            "balanced": (correct / max(len(ans), 1) + (len(una) - answered_una) / max(len(una), 1)) / 2}


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n}={k / max(n, 1):.1%} [{lo:.1%}-{hi:.1%}]"


def selector_comparison(items, label):
    ans = [i for i in items if not i["impossible"]]
    keep_all = lambda s: True
    res = {sel: {(i["scenario"], 0): float(outcome(i, keep_all, sel)[1]) for i in ans} for sel in SELECTORS}
    print(f"## {label}: answer recall with no confidence gate (n={len(ans)})")
    for sel in SELECTORS:
        k = int(sum(res[sel].values()))
        print(f"   {sel:14s} {fmt(k, len(ans))}")
    for a, b in (("lexical", "cross-encoder"), ("cross-encoder", "reader"), ("lexical", "reader")):
        d, lo, hi = paired_bootstrap(res[b], res[a], n_boot=4000)
        print(f"   {b} vs {a}: {d:+.1%} [95% CI {lo:+.1%}, {hi:+.1%}]{' *' if lo > 0 or hi < 0 else ''}")


def choose_threshold(items, selector, signal_fn, grid):
    best = None
    for t in grid:
        r = evaluate(items, selector, lambda i, t=t: (signal_fn(i) is not None and signal_fn(i) >= t))
        if best is None or r["balanced"] > best[1]["balanced"]:
            best = (t, r)
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--check", nargs="+", required=True)
    parser.add_argument("--ood", nargs="*", default=[])
    parser.add_argument("--selector", default="reader")
    args = parser.parse_args(argv)
    dev, check = load(args.dev), load(args.check)
    for p in args.dev + args.check + args.ood:
        selector_comparison(load([p]), p.split("/")[-1])

    sel = args.selector
    signals = {
        "passage logit": lambda i: (best_sub(i, sel) or {}).get("top"),
        "sentence score": lambda i: (best_sub(i, sel) or {}).get("sentence"),
        "reader margin": lambda i: (best_sub(i, sel) or {}).get("reader_margin"),
        "NLI 1-contradiction": lambda i: (None if (best_sub(i, sel) or {}).get("nli_contra_rd") is None
                                          else 1 - best_sub(i, sel)["nli_contra_rd"]),
    }
    grid = [x / 2 for x in range(-30, 31)]
    print(f"\n## refusal with selector={sel}; threshold chosen on DEV (max balanced), applied to CHECK")
    print(f"   no gate (DEV):   {evaluate(dev, sel, lambda i: True)}")
    print(f"   no gate (CHECK): {evaluate(check, sel, lambda i: True)}")
    for name, fn in signals.items():
        t, rdev = choose_threshold(dev, sel, fn, grid)
        rchk = evaluate(check, sel, lambda i, t=t: fn(i) is not None and fn(i) >= t)
        print(f"   {name:15s} t={t:+.1f}  DEV answer {fmt(*rdev['answer_recall'])} refuse {fmt(*rdev['refusal'])} "
              f"bal {rdev['balanced']:.3f} | CHECK answer {fmt(*rchk['answer_recall'])} refuse {fmt(*rchk['refusal'])} "
              f"conf-wrong {fmt(*rchk['confidently_wrong'])} bal {rchk['balanced']:.3f}")

    # logistic combinations, fitted on DEV only, compared on CHECK
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    subsets = {
        "reranker only": ["top", "sentence", "overlap_ce", "overlap_section", "margin"],
        "+ NLI": ["top", "sentence", "overlap_ce", "overlap_section", "margin", "nli_contra_rd", "nli_entail_rd"],
        "+ reader": ["top", "sentence", "overlap_ce", "overlap_section", "margin", "reader_margin"],
        "+ NLI + reader": FEATURES,
    }
    label = lambda i: int((not i["impossible"]) and outcome(i, lambda s: True, sel)[1])
    for name, cols in subsets.items():
        idx = [FEATURES.index(c) for c in cols]
        fx = lambda i: None if features(i) is None else [features(i)[k] for k in idx]
        xd = [(fx(i), i) for i in dev if fx(i) is not None]
        scaler = StandardScaler().fit([f for f, _ in xd])
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform([f for f, _ in xd]), [label(i) for _, i in xd])
        prob = lambda i, clf=clf, scaler=scaler, fx=fx: (
            None if fx(i) is None else float(clf.predict_proba(scaler.transform([fx(i)]))[0, 1]))
        xc = [(prob(i), label(i)) for i in check if prob(i) is not None]
        auc = roc_auc_score([y for _, y in xc], [p for p, _ in xc])
        t, rdev = choose_threshold(dev, sel, prob, [x / 100 for x in range(5, 96)])
        rchk = evaluate(check, sel, lambda i, t=t, prob=prob: prob(i) is not None and prob(i) >= t)
        print(f"   combo {name:15s} CHECK AUC {auc:.3f}  p>={t:.2f} | CHECK answer {fmt(*rchk['answer_recall'])} "
              f"refuse {fmt(*rchk['refusal'])} conf-wrong {fmt(*rchk['confidently_wrong'])} bal {rchk['balanced']:.3f}")
        print(f"         coefficients: {dict(zip(cols, np.round(clf.coef_[0], 2).tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
