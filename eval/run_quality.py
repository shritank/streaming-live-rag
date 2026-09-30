"""python -m eval.run_quality --scenarios DIR --corpus-dir DIR [--set a.b=v ...] [--out run.json]

Scores one configuration on one scenario set with BOTH the G2-G6 gates and
the gold-referenced answer-quality metrics (eval/quality.py), with 95% Wilson
intervals, and saves per-item outcomes so two runs can be compared on exactly
the same items (python -m eval.compare_runs a.json b.json).

`--set` overrides any config field by dotted path, e.g.
    --set retrieval.dense_encoder=e5-small-v2 --set retrieval.reranker=cross-encoder
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from streaming_rag.config import load_config
from streaming_rag.telemetry.trace import assemble_turns

from types import SimpleNamespace

from .gates import score_g2, score_g3, score_g4, score_g5, score_g6
from .loader import discover_scenarios, load_ground_truth
from .quality import (QualityResult, final_sub_query_counts, load_doc_titles, load_qrels_index,
                      score_quality, summarize)
from .runner import run_scenario_detailed
from .stats import fmt_rate

QUALITY_METRICS = ("answer", "citation", "claim_correct", "abstention", "split_exact", "oversplit")


def _raw_answers(answers: dict) -> dict:
    return {f"{sid}|{v}": {"change_kind": a.change_kind, "uncertainty": a.uncertainty,
                            "claims": [{"text": c.text, "citations": list(c.citations)} for c in a.claims]}
            for (sid, v), a in answers.items()}


def _answers_from_raw(raw: dict) -> dict:
    out = {}
    for key, a in raw.items():
        sid, _, v = key.rpartition("|")
        claims = [SimpleNamespace(text=c["text"], citations=c["citations"]) for c in a["claims"]]
        out[(sid, int(v))] = SimpleNamespace(change_kind=a["change_kind"], uncertainty=a["uncertainty"],
                                             claims=claims)
    return out


def score_raw(raw: dict, scenario_dir: str, corpus_dir: str) -> QualityResult:
    """Recompute every quality metric from saved raw outputs, without
    re-running the engine."""
    qrels, titles = load_qrels_index(corpus_dir), load_doc_titles(corpus_dir)
    quality = QualityResult()
    for path in discover_scenarios(scenario_dir):
        rec = raw.get(path.stem)
        if rec is None:
            continue
        quality.merge(score_quality(path.stem, load_ground_truth(path), rec["results"],
                                    _answers_from_raw(rec["answers"]), rec["subq_counts"], qrels, titles))
    return quality


def _attach_quality(report: dict, quality: QualityResult) -> dict:
    report["quality"] = summarize(quality)
    report["items"] = {name: {f"{k[0]}|{k[1]}": v for k, v in getattr(quality, name).items()}
                       for name in QUALITY_METRICS}
    report["wrong_claims"] = quality.wrong_claims
    return report


def apply_overrides(config, overrides: list[str]):
    for item in overrides:
        path, _, raw = item.partition("=")
        target = config
        parts = path.split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        current = getattr(target, parts[-1])
        if isinstance(current, bool):
            value = raw.lower() in ("1", "true", "yes")
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif current is None:
            value = None if raw.lower() == "none" else float(raw)
        else:
            value = raw
        setattr(target, parts[-1], value)
    return config


def _post_utterance_latencies(trace: list[dict]) -> list[float]:
    out = []
    for turn in assemble_turns(trace):
        if not turn.of("retrieval_started"):
            continue
        final = next((e for e in turn.of("chunk_received") if e.get("is_final")), None)
        output = turn.first("output_emitted")
        if final is not None and output is not None:
            out.append(max(0.0, output["ts_ms"] - final["ts_ms"]))
    return out


async def run(scenario_dir: str, corpus_dir: str, overrides: list[str], time_scale: float) -> dict:
    config = apply_overrides(load_config(), overrides)
    config.corpus_dir = corpus_dir
    gates = {"g2": [0, 0, 0, 0], "g3": [0, 0], "g4": [0, 0, 0], "g5": [0, 0], "g6": [0, 0]}
    latencies: list[float] = []
    raw: dict = {}
    started = time.perf_counter()

    for path in discover_scenarios(scenario_dir):
        gt = load_ground_truth(path)
        results, trace, answers = await run_scenario_detailed(str(path), config, time_scale=time_scale)
        g2, g3, g4, g5, g6 = (score_g2(trace, gt), score_g3(trace, gt), score_g4(trace, gt),
                              score_g5(trace, gt), score_g6(trace))
        gates["g2"] = [a + b for a, b in zip(gates["g2"], (g2.early, g2.eligible, g2.false_triggers, g2.no_retrieval_cases))]
        gates["g3"] = [a + b for a, b in zip(gates["g3"], (g3.correctly_split, g3.compound))]
        gates["g4"] = [a + b for a, b in zip(gates["g4"], (g4.n_supported, g4.n_claims, len(g4.fabricated)))]
        gates["g5"] = [a + b for a, b in zip(gates["g5"], (g5.continuous, g5.refinement_turns))]
        gates["g6"] = [a + b for a, b in zip(gates["g6"], (g6.covered_turns, g6.total_turns))]
        raw[path.stem] = {"results": results, "answers": _raw_answers(answers),
                          "subq_counts": final_sub_query_counts(trace)}
        latencies += _post_utterance_latencies(trace)

    latencies.sort()
    pct = lambda p: latencies[min(len(latencies) - 1, int(p * (len(latencies) - 1)))] if latencies else None
    from streaming_rag.retrieval.neural import ACTIVE_PROVIDERS
    report = {
        "scenario_dir": scenario_dir, "corpus_dir": corpus_dir, "overrides": overrides,
        "wall_s": round(time.perf_counter() - started, 1), "gates": gates,
        "execution_providers": {Path(k).name: v for k, v in ACTIVE_PROVIDERS.items()},
        "latency_post_utterance_ms": {"p50": pct(0.5), "p95": pct(0.95), "n": len(latencies)},
        "raw": raw,
    }
    return _attach_quality(report, score_raw(raw, scenario_dir, corpus_dir))


def print_summary(report: dict) -> None:
    g = report["gates"]
    print(f"config overrides: {report['overrides'] or '(defaults)'}   wall {report['wall_s']}s   "
          f"providers {sorted(set(report.get('execution_providers', {}).values())) or '-'}")
    print(f"  G2 early retrieval   {fmt_rate(g['g2'][0], g['g2'][1])}  false triggers {g['g2'][2]}/{g['g2'][3]}")
    print(f"  G3 >=2 sub-queries   {fmt_rate(g['g3'][0], g['g3'][1])}")
    print(f"  G4 self-reported     {fmt_rate(g['g4'][0], g['g4'][1])}  fabricated={g['g4'][2]}")
    print(f"  G5 continuity        {fmt_rate(g['g5'][0], g['g5'][1])}")
    print(f"  G6 coverage          {fmt_rate(g['g6'][0], g['g6'][1])}")
    labels = {"answer": "answer recall (gold)", "citation": "citation recall (gold)",
              "claim_correct": "claim precision (gold)", "abstention": "correct abstention",
              "split_exact": "exact decomposition", "oversplit": "over-split (lower=better)"}
    for name, label in labels.items():
        row = report["quality"][name]
        print(f"  {label:24s} {fmt_rate(row['k'], row['n'])}")
    cw = report["quality"].get("claim_words", {})
    if cw.get("mean") is not None:
        print(f"  mean words per claim     {cw['mean']:.1f} (n={cw['n']})")
    lat = report["latency_post_utterance_ms"]
    print(f"  post-utterance latency p50={lat['p50']} p95={lat['p95']} (virtual ms, n={lat['n']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios")
    parser.add_argument("--corpus-dir")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--time-scale", type=float, default=8.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--rescore", default=None, help="recompute metrics for a saved run JSON in place")
    args = parser.parse_args(argv)
    if args.rescore:
        report = json.loads(Path(args.rescore).read_text(encoding="utf-8"))
        report = _attach_quality(report, score_raw(report["raw"], report["scenario_dir"], report["corpus_dir"]))
        Path(args.rescore).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print_summary(report)
        return 0
    report = asyncio.run(run(args.scenarios, args.corpus_dir, args.overrides, args.time_scale))
    print_summary(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
