"""python -m eval.refusal_policy --dev A.json B.json --check C.json D.json --ood E.json

Decision analysis for the refusal gate. For each feature set, a logistic
model of P(the answer we would give is correct) is fitted on DEV. For each
deployment assumption (share of unanswerable questions p_una, and cost c of
a confidently wrong answer relative to the value of a correct one) the
threshold maximising expected utility on DEV is chosen:

    U = (1 - p_una) * (answer_recall - c * wrong_rate_on_answerable)
        - p_una * c * answered_rate_on_unanswerable

and then applied unchanged to CHECK and to the out-of-distribution set
(OOD: all answerable, so the gate can only cost recall there).
"""
from __future__ import annotations

import argparse

import numpy as np

from .analyze_sweeps import FEATURES, features, load
from .refusal_sweep import outcome
from .stats import wilson

SUBSETS = {
    "none (no gate)": None,
    "reranker only": ["top", "sentence", "overlap_ce", "overlap_section", "margin"],
    "+ NLI": ["top", "sentence", "overlap_ce", "overlap_section", "margin", "nli_contra_rd", "nli_entail_rd"],
    "+ reader": ["top", "sentence", "overlap_ce", "overlap_section", "margin", "reader_margin"],
    "+ NLI + reader": FEATURES,
    # only features whose direction is causally sound off SQuAD2: a better
    # claim-question match and a reader answer beating no-answer both mean
    # "more likely correct" (retrieval score and query-passage overlap are
    # negatively weighted by the 6-feature fit, an artefact of how SQuAD2's
    # unanswerable questions were written: by copying passage words)
    "sentence + reader": ["sentence", "reader_margin"],
    # + whether the reader's answer span lies inside the CE-selected sentence
    "+ reader + agree": ["top", "sentence", "overlap_ce", "overlap_section", "margin", "reader_margin", "agree"],
    "reader only": ["reader_margin"],
    "sentence only": ["sentence"],
}
SCENARIOS = [(0.1, 1.0), (0.1, 2.0), (0.3, 1.0), (0.3, 2.0), (0.5, 1.0), (0.5, 2.0)]
SEL = "cross-encoder"


_OUTCOME: dict[int, tuple[bool, bool]] = {}


def _out(i):
    if id(i) not in _OUTCOME:
        _OUTCOME[id(i)] = outcome(i, lambda s: True, SEL)
    return _OUTCOME[id(i)]


def rates(items, answer_fn):
    ans = [i for i in items if not i["impossible"]]
    una = [i for i in items if i["impossible"]]
    r = w = a = 0
    for i in ans:
        if answer_fn(i):
            asserted, correct = _out(i)
            r += correct
            w += asserted and not correct
    for i in una:
        if answer_fn(i):
            a += _out(i)[0]
    return {"R": r / max(len(ans), 1), "W": w / max(len(ans), 1), "A": a / max(len(una), 1),
            "n_ans": len(ans), "n_una": len(una), "r": r, "w": w, "a": a}


def utility(m, p_una, c):
    return (1 - p_una) * (m["R"] - c * m["W"]) - p_una * c * m["A"]


def fit(dev, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    idx = [FEATURES.index(c) for c in cols]
    fx = lambda i: None if features(i) is None else [features(i)[k] for k in idx]
    rows = [(fx(i), i) for i in dev if fx(i) is not None]
    y = [int((not i["impossible"]) and outcome(i, lambda s: True, SEL)[1]) for _, i in rows]
    scaler = StandardScaler().fit([f for f, _ in rows])
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform([f for f, _ in rows]), y)
    cache: dict[int, float | None] = {}

    def prob(i):
        if id(i) not in cache:
            f = fx(i)
            cache[id(i)] = None if f is None else float(clf.predict_proba(scaler.transform([f]))[0, 1])
        return cache[id(i)]
    prob.model = (cols, scaler, clf)
    return prob


def export(args, dev, grid) -> int:
    import json
    from pathlib import Path
    cols = SUBSETS[args.export]
    prob = fit(dev, cols)
    gate = lambda t: (lambda i: prob(i) is not None and prob(i) >= t)
    t = max(grid, key=lambda t: utility(rates(dev, gate(t)), args.p_una, args.cost))
    _, scaler, clf = prob.model
    model = {
        "description": "Refusal gate: logistic model of P(the asserted answer is correct). Refuse when p < threshold.",
        "features": cols,
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "coef": [float(x) for x in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "threshold": t,
        "fitted_on": args.dev,
        "threshold_rule": f"maximise expected utility on DEV with p_unanswerable={args.p_una}, cost of a wrong "
                          f"answer c={args.cost} (U = (1-p)(recall - c*wrong_rate) - p*c*answered_unanswerable_rate)",
    }
    Path(args.out).write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {cols} threshold={t}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--check", nargs="+", required=True)
    parser.add_argument("--ood", nargs="*", default=[])
    parser.add_argument("--export", default=None, help="SUBSETS name to write as a production gate JSON")
    parser.add_argument("--p-una", type=float, default=0.3)
    parser.add_argument("--cost", type=float, default=1.0)
    parser.add_argument("--out", default="streaming_rag/session/refusal_gate.json")
    args = parser.parse_args(argv)
    dev, check, ood = load(args.dev), load(args.check), load(args.ood) if args.ood else []
    grid = [x / 100 for x in range(0, 100, 2)]
    if args.export:
        return export(args, dev, grid)
    for name, cols in SUBSETS.items():
        prob = fit(dev, cols) if cols else None
        gate = (lambda t: (lambda i: True)) if prob is None else \
            (lambda t: (lambda i: prob(i) is not None and prob(i) >= t))
        print(f"## gate features: {name}")
        if prob is not None:
            cols_, scaler, clf = prob.model
            print("   coef " + ", ".join(f"{c}={w:+.3f}" for c, w in zip(cols_, clf.coef_[0]))
                  + f", intercept={clf.intercept_[0]:+.3f}")
        for p_una, c in SCENARIOS:
            t = 0.0 if prob is None else max(grid, key=lambda t: utility(rates(dev, gate(t)), p_una, c))
            m = rates(check, gate(t))
            o = rates(ood, gate(t)) if ood else None
            o0 = rates(ood, lambda i: True) if ood else None
            lo, hi = wilson(m["r"], m["n_ans"])
            print(f"   p_una={p_una:.1f} c={c:.0f}: t={t:.2f} | CHECK answer {m['R']:.1%} [{lo:.1%}-{hi:.1%}] "
                  f"wrong-on-answerable {m['W']:.1%} answered-unanswerable {m['A']:.1%} "
                  f"U={utility(m, p_una, c):+.3f}"
                  + (f" | OOD answer {o['R']:.1%} (no gate {o0['R']:.1%})" if o else ""))
            if prob is None:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
