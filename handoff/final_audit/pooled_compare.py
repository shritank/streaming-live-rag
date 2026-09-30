"""Pooled paired comparison of two configurations over several evaluation sets.

    python handoff/final_audit/pooled_compare.py "A1.json,A2" "B1.json,B2" [label]

Each side is a comma-separated list of run_quality outputs on the SAME scenario sets (same order). Items are
pooled (keys are prefixed with the set index so clusters stay distinct) and compared with the project's
cluster-bootstrap (eval.stats.paired_bootstrap; 4000 resamples, clustered by scenario). Used to apply the
"DEV decides, DIAG confirms" rule when each single set is too small to be significant on its own.
"""
import json
import sys

sys.path.insert(0, ".")
from eval.compare_runs import METRICS, _items  # noqa: E402
from eval.stats import paired_bootstrap  # noqa: E402


def pooled(paths, metric):
    out = {}
    for i, p in enumerate(paths):
        rep = json.loads(open(p, encoding="utf-8").read())
        for (scen, item), v in _items(rep, metric).items():
            out[(f"{i}:{scen}", item)] = v
    return out


def main():
    a = sys.argv[1].split(",")
    b = sys.argv[2].split(",")
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    print(f"pooled over {len(a)} sets {label}")
    for m in METRICS:
        ia, ib = pooled(a, m), pooled(b, m)
        keys = set(ia) & set(ib)
        if not keys:
            continue
        diff, lo, hi = paired_bootstrap(ib, ia)
        star = "*" if lo > 0 or hi < 0 else " "
        print(f"  {m:14s} {sum(ia[k] for k in keys) / len(keys):6.1%} -> {sum(ib[k] for k in keys) / len(keys):6.1%}  "
              f"diff {diff:+.1%} [95% CI {lo:+.1%}, {hi:+.1%}] {star} (n={len(keys)})")


if __name__ == "__main__":
    main()
