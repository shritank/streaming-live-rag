"""Small-sample statistics for evaluation results.

Every rate this project reports comes from a few dozen items, so a point
estimate alone overstates what is known. Two tools:

  wilson(k, n)              95% Wilson score interval for a proportion
                            (well-behaved at k=0, k=n and small n, unlike the
                            normal approximation)
  paired_bootstrap(a, b)    95% percentile CI for mean(a) - mean(b) when a and
                            b are outcomes of two systems on the SAME items,
                            resampled by cluster (scenario) so correlated turns
                            inside one scenario are not treated as independent
"""
from __future__ import annotations

import math
import random
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.1%} [95% CI {lo:.1%}-{hi:.1%}]"


def paired_bootstrap(items_a: dict[tuple, float], items_b: dict[tuple, float],
                     n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """items_* map (cluster_id, item_id) -> outcome in [0, 1]. Only items present
    in both systems are compared. Returns (observed_diff, ci_low, ci_high) for
    mean(a) - mean(b), resampling whole clusters with replacement."""
    keys = sorted(set(items_a) & set(items_b))
    if not keys:
        return (0.0, 0.0, 0.0)
    clusters: dict = defaultdict(list)
    for key in keys:
        clusters[key[0]].append(items_a[key] - items_b[key])
    cluster_ids = sorted(clusters)
    observed = sum(d for c in cluster_ids for d in clusters[c]) / len(keys)

    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        total = count = 0.0
        for _ in cluster_ids:
            values = clusters[cluster_ids[rng.randrange(len(cluster_ids))]]
            total += sum(values)
            count += len(values)
        diffs.append(total / count if count else 0.0)
    diffs.sort()
    lo = diffs[int(0.025 * (n_boot - 1))]
    hi = diffs[int(0.975 * (n_boot - 1))]
    return (observed, lo, hi)
