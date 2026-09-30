"""python -m eval.ablate [--scenarios DIR] [--corpus-dir DIR]

Ablations of the final system (project context section 13 asks for >= 2), scored with the
gold-referenced metrics from eval/quality.py rather than G4: G4 is the synthesizer grading its own
extractive claims, which is ~100% for every configuration and so cannot tell them apart.

  1. retrieval.mode     hybrid vs sparse-only vs dense-only (same reranker)
  2. retrieval.reranker cross-encoder rerank of the fused candidates vs none
  3. synthesis          learned refusal gate vs none; cross-encoder claim selection vs lexical
  4. engine             speculative final-pass retrieval on vs off (answers must be identical)
  controller.mode=model (rule vs model) is NOT implemented: build.py rejects it, and it would need an
  LLM key. It is reported as such, never as a measurement.

Paired significance for any two of the saved runs: python -m eval.compare_runs A.json B.json
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

    def run_cfg(name: str, overrides: list[str]) -> None:
        _print(name, asyncio.run(run(args.scenarios, args.corpus_dir, overrides, args.time_scale)))

    print("--- Ablation 1: retrieval.mode ---")
    for mode in ("hybrid", "sparse", "dense"):
        run_cfg(f"retrieval.mode={mode}", [f"retrieval.mode={mode}"])
    print("--- Ablation 2: cross-encoder rerank of the fused candidates ---")
    run_cfg("retrieval.reranker=cross-encoder (default)", [])
    run_cfg("retrieval.reranker=none", ["retrieval.reranker=none"])
    print("--- Ablation 3: synthesis ---")
    run_cfg("refusal_gate=learned (default)", [])
    run_cfg("refusal_gate=none", ["synthesis.refusal_gate=none"])
    run_cfg("selector=lexical", ["synthesis.selector=lexical"])
    print("--- Ablation 4: engine ---")
    run_cfg("speculative_final=false", ["engine.speculative_final=false"])
    print("--- controller.mode: rule vs model ---")
    print("  not implemented (controller.mode=model is rejected at build time); not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
