"""python -m eval.retrieval_eval --corpus-dir DIR [--sample N] [--config name:k=v,k=v ...]

Standalone retrieval benchmark against a corpus's qrels.jsonl: recall@1/5/10,
MRR@10 and nDCG@10 (one gold section per query, binary relevance), with 95%
Wilson intervals, for one or more retrieval configurations evaluated on the
SAME sampled queries so they can be compared pairwise (per-query outcomes are
compared with a paired bootstrap; clusters = queries).

    python -m eval.retrieval_eval --corpus-dir data/corpus --sample 1000 \
        --config baseline: \
        --config e5_ce:retrieval.dense_encoder=e5-small-v2,retrieval.reranker=cross-encoder
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

from streaming_rag.config import load_config
from streaming_rag.contracts import SubQuery
from streaming_rag.retrieval import HybridRetriever

from .run_quality import apply_overrides
from .stats import paired_bootstrap, wilson


def load_qrels(corpus_dir: str) -> list[dict]:
    lines = Path(corpus_dir, "qrels.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def evaluate(corpus_dir: str, overrides: list[str], queries: list[dict], k: int = 10) -> dict:
    config = apply_overrides(load_config(), overrides)
    config.corpus_dir = corpus_dir
    retriever = HybridRetriever(config)
    asyncio.run(retriever.setup())
    per_query = []
    doc_hits: list[bool] = []
    started = time.perf_counter()
    for n, q in enumerate(queries):
        evidence = retriever._search_sync(SubQuery(query_id=f"q{n}", text=q["query"], intent_label="",
                                                     trigger="final", utterance_id="eval"), k)
        cites = [e.chunk.citation for e in evidence]
        rank = next((i + 1 for i, c in enumerate(cites) if c in q["relevant"]), None)
        per_query.append(rank)
        gold_doc = q["relevant"][0].split(" §")[0]
        doc_hits.append(bool(evidence) and evidence[0].chunk.doc_id == gold_doc)
    elapsed = time.perf_counter() - started
    n = len(per_query)
    out = {"overrides": overrides, "n": n, "ms_per_query": 1000 * elapsed / max(n, 1), "ranks": per_query}
    for cut in (1, 5, 10):
        hits = sum(1 for r in per_query if r is not None and r <= cut)
        out[f"r@{cut}"] = (hits, n)
    out["doc_r@1"] = (sum(doc_hits), n)
    out["mrr@10"] = sum(1 / r for r in per_query if r is not None and r <= 10) / n
    out["ndcg@10"] = sum(1 / math.log2(r + 1) for r in per_query if r is not None and r <= 10) / n
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--sample", type=int, default=0, help="0 = all qrels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", action="append", default=[],
                        help="name:dotted.key=value,dotted.key=value (empty after ':' = defaults)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--qrels", default=None, help="alternative qrels JSONL (default: <corpus-dir>/qrels.jsonl)")
    args = parser.parse_args(argv)

    if args.qrels:
        lines = Path(args.qrels).read_text(encoding="utf-8").splitlines()
        qrels = [json.loads(l) for l in lines if l.strip()]
    else:
        qrels = load_qrels(args.corpus_dir)
    if args.sample and args.sample < len(qrels):
        qrels = random.Random(args.seed).sample(qrels, args.sample)
    configs = args.config or ["baseline:"]
    results = {}
    for spec in configs:
        name, _, rest = spec.partition(":")
        overrides = [kv for kv in rest.split(",") if kv]
        results[name] = evaluate(args.corpus_dir, overrides, qrels)

    base_name = next(iter(results))
    for name, r in results.items():
        cells = []
        for cut in (1, 5, 10):
            h, n = r[f"r@{cut}"]
            lo, hi = wilson(h, n)
            cells.append(f"r@{cut} {h / n:.1%} [{lo:.1%}-{hi:.1%}]")
        dh, dn = r["doc_r@1"]
        print(f"{name:14s} n={r['n']}  " + "  ".join(cells) +
              f"  MRR@10 {r['mrr@10']:.3f}  nDCG@10 {r['ndcg@10']:.3f}  doc r@1 {dh / dn:.1%}"
              f"  {r['ms_per_query']:.1f} ms/query")
        if name != base_name:
            base = results[base_name]["ranks"]
            for cut in (1, 5):
                a = {(i, 0): float(x is not None and x <= cut) for i, x in enumerate(base)}
                b = {(i, 0): float(x is not None and x <= cut) for i, x in enumerate(r["ranks"])}
                d, lo, hi = paired_bootstrap(b, a, n_boot=2000)
                print(f"{'':14s}   vs {base_name}: r@{cut} {d:+.1%} [95% CI {lo:+.1%}, {hi:+.1%}]"
                      f"{'  *' if lo > 0 or hi < 0 else ''}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
