"""python -m eval.run_all [--scenarios DIR] [--time-scale 8] [--reps 3] [--impl real]

Runs every scenario 3 times (fresh engine instances), takes the median trace,
scores gates G2-G6 against the aggregated result, and prints VERDICT: PASS/FAIL.
G1 (reproducibility) is a separate, external check (docker compose up on a
clean machine) and is not scored here.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from streaming_rag.config import load_config

from .gates import score_g2, score_g3, score_g4, score_g5, score_g6
from .gates.g6_coverage import G6Result
from .loader import discover_scenarios, load_ground_truth
from .runner import run_scenario_median
from .schema_check import load_schema, validate_events

THRESHOLDS = {
    "G2_early_rate": 0.80, "G2_false_trigger_rate": 0.10,
    "G3_rate": 0.70, "G4_support_rate": 0.85, "G5_rate": 1.00, "G6_coverage": 1.00,
}


async def run_suite(scenario_dir: str, time_scale: float, reps: int, impl: str, corpus_dir: str | None):
    config = load_config()
    if corpus_dir:
        config.corpus_dir = corpus_dir

    scenarios = discover_scenarios(scenario_dir)
    if not scenarios:
        raise SystemExit(f"no scenarios found in {scenario_dir}")

    g2_total = g2_early = g2_no_retrieval = g2_false = 0
    g3_compound = g3_split = 0
    g4_claims = g4_supported = 0
    g4_fabricated: list[str] = []
    g5_turns = g5_ok = 0
    g5_failures: list[str] = []
    g6_total_turns = g6_covered_turns = 0
    g6_problems: list[str] = []
    schema_errors: list[str] = []
    per_scenario = []

    schema = load_schema()

    for path in scenarios:
        ground_truth = load_ground_truth(path)
        turn_results, trace = await run_scenario_median(str(path), config, impl=impl, time_scale=time_scale, reps=reps)

        schema_errors.extend(f"{path.name}: {e}" for e in validate_events(trace, schema))

        g2 = score_g2(trace, ground_truth)
        g3 = score_g3(trace, ground_truth)
        g4 = score_g4(trace, ground_truth)
        g5 = score_g5(trace, ground_truth)
        # Scored per-scenario, not on a concatenated trace: scenarios reuse
        # session/utterance ids like "s1"/"u1", so merging traces across
        # scenarios before grouping into turns would corrupt turn boundaries.
        g6 = score_g6(trace)

        g2_total += g2.eligible; g2_early += g2.early
        g2_no_retrieval += g2.no_retrieval_cases; g2_false += g2.false_triggers
        g3_compound += g3.compound; g3_split += g3.correctly_split
        g4_claims += g4.n_claims; g4_supported += g4.n_supported; g4_fabricated += g4.fabricated
        g5_turns += g5.refinement_turns; g5_ok += g5.continuous; g5_failures += g5.failures
        g6_total_turns += g6.total_turns; g6_covered_turns += g6.covered_turns
        g6_problems += [f"{path.name}: {p}" for p in g6.problems]

        per_scenario.append((path.name, len(turn_results), len(trace)))

    g6 = G6Result(total_turns=g6_total_turns, covered_turns=g6_covered_turns, problems=g6_problems)

    report = {
        "n_scenarios": len(scenarios),
        "G2": {"eligible": g2_total, "early": g2_early,
               "early_rate": g2_early / g2_total if g2_total else 1.0,
               "no_retrieval_cases": g2_no_retrieval, "false_triggers": g2_false,
               "false_trigger_rate": g2_false / g2_no_retrieval if g2_no_retrieval else 0.0},
        "G3": {"compound": g3_compound, "correctly_split": g3_split,
               "rate": g3_split / g3_compound if g3_compound else 1.0},
        "G4": {"n_claims": g4_claims, "n_supported": g4_supported,
               "support_rate": g4_supported / g4_claims if g4_claims else 1.0,
               "fabricated": g4_fabricated},
        "G5": {"refinement_turns": g5_turns, "continuous": g5_ok,
               "rate": g5_ok / g5_turns if g5_turns else 1.0, "failures": g5_failures},
        "G6": {"total_turns": g6.total_turns, "covered_turns": g6.covered_turns,
               "coverage_rate": g6.coverage_rate, "problems": g6.problems[:20],
               "schema_errors": schema_errors[:20]},
        "per_scenario": per_scenario,
    }
    return report


def verdict(report: dict) -> tuple[bool, list[str]]:
    checks = [
        ("G2 early_rate >= 80%", report["G2"]["early_rate"] >= THRESHOLDS["G2_early_rate"]),
        ("G2 false_trigger_rate <= 10%", report["G2"]["false_trigger_rate"] <= THRESHOLDS["G2_false_trigger_rate"]),
        ("G3 rate >= 70%", report["G3"]["rate"] >= THRESHOLDS["G3_rate"]),
        ("G4 support_rate >= 85%", report["G4"]["support_rate"] >= THRESHOLDS["G4_support_rate"]),
        ("G4 zero fabricated citations", len(report["G4"]["fabricated"]) == 0),
        ("G5 rate == 100%", report["G5"]["rate"] >= THRESHOLDS["G5_rate"]),
        ("G6 coverage == 100%", report["G6"]["coverage_rate"] >= THRESHOLDS["G6_coverage"]),
        ("G6 schema valid", len(report["G6"]["schema_errors"]) == 0),
    ]
    failed = [name for name, ok in checks if not ok]
    return len(failed) == 0, failed


def print_report(report: dict) -> None:
    print(f"scenarios run: {report['n_scenarios']}")
    print(f"G2 early retrieval : {report['G2']['early']}/{report['G2']['eligible']} "
          f"= {report['G2']['early_rate']:.1%}  (false-trigger rate {report['G2']['false_trigger_rate']:.1%})")
    print(f"G3 multi-intent    : {report['G3']['correctly_split']}/{report['G3']['compound']} "
          f"= {report['G3']['rate']:.1%}")
    print(f"G4 grounding       : {report['G4']['n_supported']}/{report['G4']['n_claims']} "
          f"= {report['G4']['support_rate']:.1%}  (fabricated={len(report['G4']['fabricated'])})")
    print(f"G5 refinement      : {report['G5']['continuous']}/{report['G5']['refinement_turns']} "
          f"= {report['G5']['rate']:.1%}")
    print(f"G6 telemetry cov.  : {report['G6']['covered_turns']}/{report['G6']['total_turns']} "
          f"= {report['G6']['coverage_rate']:.1%}  (schema errors={len(report['G6']['schema_errors'])})")
    if report["G5"]["failures"]:
        print("G5 failures:")
        for f in report["G5"]["failures"][:10]:
            print(f"   - {f}")
    if report["G6"]["problems"]:
        print("G6 problems:")
        for p in report["G6"]["problems"][:10]:
            print(f"   - {p}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="eval/scenarios")
    parser.add_argument("--time-scale", type=float, default=8.0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--impl", choices=["real", "mock"], default="real")
    parser.add_argument("--corpus-dir", default=None)
    args = parser.parse_args(argv)

    report = asyncio.run(run_suite(args.scenarios, args.time_scale, args.reps, args.impl, args.corpus_dir))
    print_report(report)
    ok, failed = verdict(report)
    print()
    print("VERDICT:", "PASS" if ok else "FAIL")
    if failed:
        print("failed checks:", "; ".join(failed))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
