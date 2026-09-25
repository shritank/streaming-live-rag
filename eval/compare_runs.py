"""python -m eval.compare_runs baseline.json candidate.json

Paired comparison of two eval.run_quality outputs on the same scenario set:
for each metric, the difference (candidate - baseline) over the items both
runs scored, with a 95% cluster-bootstrap CI (clusters = scenarios). A
difference whose CI excludes 0 is reported as significant. Claim precision
is compared as per-turn precision over turns where both systems asserted
claims (reported as claim_correct, "turn-avg").
"""
from __future__ import annotations

import argparse
import json

from .stats import paired_bootstrap

METRICS = ("answer", "citation", "claim_correct", "abstention", "split_exact", "oversplit")


def _items(report: dict, name: str) -> dict:
    items = {tuple(k.split("|", 1)): v for k, v in report["items"][name].items()}
    if name != "claim_correct":
        return items
    # Two systems' j-th claims in a turn are not the same claim, so claims
    # cannot be paired one-to-one; pair per turn on that turn's precision.
    turns: dict = {}
    for (scenario, item), v in items.items():
        turns.setdefault((scenario, item.split(":c")[0]), []).append(v)
    return {k: sum(v) / len(v) for k, v in turns.items()}


def compare(a: dict, b: dict) -> list[dict]:
    rows = []
    for name in METRICS:
        ia, ib = _items(a, name), _items(b, name)
        common = set(ia) & set(ib)
        diff, lo, hi = paired_bootstrap(ib, ia)
        rows.append({"metric": name, "n_paired": len(common),
                     "baseline": sum(ia[k] for k in common) / len(common) if common else None,
                     "candidate": sum(ib[k] for k in common) / len(common) if common else None,
                     "diff": diff, "ci": [lo, hi], "significant": lo > 0 or hi < 0})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args(argv)
    a = json.load(open(args.baseline, encoding="utf-8"))
    b = json.load(open(args.candidate, encoding="utf-8"))
    print(f"baseline : {a['overrides'] or '(defaults)'}")
    print(f"candidate: {b['overrides'] or '(defaults)'}")
    for r in compare(a, b):
        if r["baseline"] is None:
            continue
        sig = "*" if r["significant"] else " "
        label = "claim (turn-avg)" if r["metric"] == "claim_correct" else r["metric"]
        print(f"  {label:16s} {r['baseline']:.1%} -> {r['candidate']:.1%}  "
              f"diff {r['diff']:+.1%} [95% CI {r['ci'][0]:+.1%}, {r['ci'][1]:+.1%}] {sig} (n={r['n_paired']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
