"""python -m eval.ablate [--scenarios DIR] [--corpus-dir DIR]

The two required ablations (project context §13), scored with the
gold-referenced metrics from eval/quality.py rather than G4: G4 is the
synthesizer grading its own extractive claims, which is ~100% for every
retrieval mode and so cannot tell the modes apart.

  retrieval.mode   hybrid vs sparse-only vs dense-only (same reranker)
  controller.mode  rule vs model — "model" needs LLM_PROVIDER != fake; with
                   the default offline fake provider it is reported, not run
"""
from __future__ import annotations

import argparse
import asyncio

from .run_quality import run
from .stats import fmt_rate

METRICS = (("answer", "answer recall"), ("citation", "citation recall"),
           ("claim_correct", "claim precision"), ("abstention", "abstention"))


def _print(name: str, report: dict) -> None:
    print(f"  {name}")
    for key, label in METRICS:
        row = report["quality"][key]
        print(f"      {label:16s} {fmt_rate(row['k'], row['n'])}")
    lat = report["latency_post_utterance_ms"]
    print(f"      post-utterance latency p50={lat['p50']:.0f} ms p95={lat['p95']:.0f} ms")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="eval/scenarios_real_dev")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--time-scale", type=float, default=8.0)
    args = parser.parse_args(argv)

    print("--- Ablation 1: retrieval.mode ---")
    for mode in ("hybrid", "sparse", "dense"):
        report = asyncio.run(run(args.scenarios, args.corpus_dir, [f"retrieval.mode={mode}"], args.time_scale))
        _print(f"retrieval.mode={mode}", report)
    print("--- Ablation 2: controller.mode ---")
    report = asyncio.run(run(args.scenarios, args.corpus_dir, ["controller.mode=rule"], args.time_scale))
    _print("controller.mode=rule", report)
    print("  controller.mode=model: requires LLM_PROVIDER != fake (not run offline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
