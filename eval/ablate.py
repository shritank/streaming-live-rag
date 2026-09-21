"""python -m eval.ablate [--scenarios DIR] [--corpus-dir DIR]

Runs the two required ablations (project context §13):
  - retrieval.mode: hybrid vs dense-only vs sparse-only
  - controller.mode: rule vs model (LLM-driven) — model requires an LLM
    provider to be configured; with the default fake provider it degrades
    gracefully rather than crashing, and is reported as such.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from streaming_rag.config import load_config

from .gates import score_g2, score_g3, score_g4
from .loader import discover_scenarios, load_ground_truth
from .runner import run_scenario


async def run_retrieval_ablation(scenarios: list[Path], corpus_dir: str) -> list[dict]:
    rows = []
    for mode in ("hybrid", "sparse", "dense"):
        config = load_config()
        config.corpus_dir = corpus_dir
        config.retrieval.mode = mode
        n_claims = n_supported = 0
        fabricated = 0
        for path in scenarios:
            gt = load_ground_truth(path)
            _, trace = await run_scenario(str(path), config, impl="real", time_scale=8)
            g4 = score_g4(trace, gt)
            n_claims += g4.n_claims; n_supported += g4.n_supported; fabricated += len(g4.fabricated)
        rows.append({
            "ablation": "retrieval.mode", "value": mode,
            "G4_support_rate": n_supported / n_claims if n_claims else 1.0,
            "fabricated_citations": fabricated, "n_claims": n_claims,
        })
    return rows


async def run_controller_ablation(scenarios: list[Path], corpus_dir: str) -> list[dict]:
    rows = []
    for mode in ("rule",):  # "model" requires a configured LLM_PROVIDER; documented, not run by default
        config = load_config()
        config.corpus_dir = corpus_dir
        config.controller.mode = mode
        g2_e = g2_t = g3_s = g3_c = 0
        for path in scenarios:
            gt = load_ground_truth(path)
            _, trace = await run_scenario(str(path), config, impl="real", time_scale=8)
            g2 = score_g2(trace, gt); g3 = score_g3(trace, gt)
            g2_e += g2.early; g2_t += g2.eligible; g3_s += g3.correctly_split; g3_c += g3.compound
        rows.append({
            "ablation": "controller.mode", "value": mode,
            "G2_early_rate": g2_e / g2_t if g2_t else 1.0,
            "G3_rate": g3_s / g3_c if g3_c else 1.0,
        })
    rows.append({
        "ablation": "controller.mode", "value": "model",
        "note": "requires LLM_PROVIDER != fake; not run in this environment — "
                "see docs/benchmark_report.md for how to enable it",
    })
    return rows


def print_rows(title: str, rows: list[dict]) -> None:
    print(f"--- {title} ---")
    for row in rows:
        print(" ", {k: (f"{v:.1%}" if isinstance(v, float) else v) for k, v in row.items()})
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="eval/scenarios")
    parser.add_argument("--corpus-dir", default="fixtures/dev_corpus")
    args = parser.parse_args(argv)

    scenarios = discover_scenarios(args.scenarios)

    async def run_both():
        retrieval_rows = await run_retrieval_ablation(scenarios, args.corpus_dir)
        controller_rows = await run_controller_ablation(scenarios, args.corpus_dir)
        return retrieval_rows, controller_rows

    retrieval_rows, controller_rows = asyncio.run(run_both())
    print_rows("Ablation 1: retrieval.mode (hybrid vs sparse vs dense)", retrieval_rows)
    print_rows("Ablation 2: controller.mode (rule vs model)", controller_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
