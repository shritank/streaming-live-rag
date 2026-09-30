"""python -m eval.latency_profile --scenarios DIR --corpus-dir DIR [--time-scale 1] [--set k=v ...]
                                 [--out FILE]

Stage-by-stage latency profile of the unmodified streaming engine replaying
real scenarios. All times are time.perf_counter() (sub-microsecond; the
Windows time.monotonic() clock ticks every ~16 ms and cannot resolve a 5 ms
target). Nothing is excluded: every call during the measured replay is
counted. One warm-up scenario is replayed first and not counted (model and
index loading); its cost is reported separately as `cold_start_s`.

Measurement boundaries (each reported separately, never summed):

  engine.post_speech_ms        utterance_end envelope handled -> turn output
                               emitted (what the user waits after they stop
                               talking). From output_emitted.timings.
  engine.final_decision_ms     controller pass on the final (empty) chunk
  engine.retrieval_wait_ms     waiting for retrievals still in flight at
                               utterance end
  engine.synthesis_ms          synthesize()/refine() call at utterance end
  engine.answer_telemetry_ms   grounding telemetry + answer bookkeeping
  controller.decision_ms       one on_chunk() decision (every chunk)
  retrieval.subquery_ms        one sub-query's search, end to end (runs
                               concurrently with speech when speculative)
  retrieval.{bm25,e5_encode,dense_topk,rrf}_ms, ce.rerank_ms
                               stages inside one search
  synthesis.claim_select_ms, synthesis.gate_ms, ce.claim_ms, ce.gate_ms,
  reader.gate_ms, synthesis.grounding_ms, synthesis.prefetch_ms
                               stages inside claim selection
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from streaming_rag.config import load_config
from streaming_rag.retrieval import bm25 as bm25_mod
from streaming_rag.retrieval import hybrid, neural
from streaming_rag.session import synthesis as synth_mod

from .loader import discover_scenarios
from .run_quality import apply_overrides
from .runner import run_scenario_detailed

SAMPLES: dict[str, list[float]] = defaultdict(list)
RECORDING = threading.Event()
_ctx = threading.local()


CURRENT = {"scenario": None, "t0": 0.0}          # where the slowest samples happened
WORST: dict[str, list[tuple[float, str, float]]] = defaultdict(list)
TAIL: list[dict] = []                               # every turn with post-speech > 5 ms
SPEC = {"launched": 0, "reused": 0}                  # speculative final-pass retrievals


def _record(name: str, ms: float) -> None:
    if RECORDING.is_set():
        SAMPLES[name].append(ms)
        w = WORST[name]
        w.append((ms, CURRENT["scenario"], round(time.perf_counter() - CURRENT["t0"], 2)))
        w.sort(reverse=True)
        del w[5:]


def _timed(name: str | None, fn, ctx: str | None = None, name_from_ctx: str | None = None):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        prev = getattr(_ctx, "tag", None)
        if ctx:
            _ctx.tag = ctx
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            ms = (time.perf_counter() - t) * 1000
            if ctx:
                _ctx.tag = prev
            key = name or f"{name_from_ctx}.{getattr(_ctx, 'tag', None) or 'other'}_ms"
            _record(key, ms)
    return wrapper


def _timed_async(name: str, fn):
    @functools.wraps(fn)
    async def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return await fn(*a, **kw)
        finally:
            _record(name, (time.perf_counter() - t) * 1000)
    return wrapper


def install() -> None:
    bm25_mod.BM25Index.top_k = _timed("retrieval.bm25_ms", bm25_mod.BM25Index.top_k)
    hybrid.cosine_top_k = _timed("retrieval.dense_topk_ms", hybrid.cosine_top_k)
    hybrid.rrf = _timed("retrieval.rrf_ms", hybrid.rrf)
    hybrid.HybridRetriever._search_sync = _timed(None, hybrid.HybridRetriever._search_sync, ctx="rerank",
                                                 name_from_ctx="retrieval.search")
    enc = neural.E5Embedder.encode

    def encode(self, texts):          # query encodes only (1 text); corpus encodes are setup, not per query
        if len(texts) > 4:
            return enc(self, texts)
        return _timed("retrieval.e5_encode_ms", enc)(self, texts)
    neural.E5Embedder.encode = encode
    neural.CrossEncoder.score = _timed(None, neural.CrossEncoder.score, name_from_ctx="ce")
    neural.SquadReader.answer = _timed(None, neural.SquadReader.answer, name_from_ctx="reader")
    G = synth_mod.GroundedSynthesizer
    G._ce_claim = _timed("synthesis.claim_select_ms", G._ce_claim, ctx="claim")
    G._answer_probability = _timed("synthesis.gate_ms", G._answer_probability, ctx="gate")
    G.prefetch = _timed("synthesis.prefetch_ms", G.prefetch)
    synth_mod.grounding.verify = _timed("synthesis.grounding_ms", synth_mod.grounding.verify)


def summarise(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 3),
            "p95": round(float(np.percentile(a, 95)), 3), "p99": round(float(np.percentile(a, 99)), 3),
            "max": round(float(a.max()), 3), "mean": round(float(a.mean()), 3)}


async def run(scenarios: str, corpus_dir: str, overrides: list[str], time_scale: float) -> dict:
    config = apply_overrides(load_config(), overrides)
    config.corpus_dir = corpus_dir
    paths = discover_scenarios(scenarios)
    t = time.perf_counter()
    await run_scenario_detailed(str(paths[0]), config, time_scale=time_scale)     # warm-up, not counted
    cold = time.perf_counter() - t
    RECORDING.set()
    for p in paths:
        CURRENT["scenario"], CURRENT["t0"] = p.stem, time.perf_counter()
        _, trace, _ = await run_scenario_detailed(str(p), config, time_scale=time_scale)
        ends = {e["utterance_id"]: e["ts_ms"] for e in trace
                if e["event"] == "chunk_received" and e.get("is_final")}
        for e in trace:
            if e["event"] == "controller_decision":
                SAMPLES["controller.decision_ms"].append(e["decision_latency_ms"])
            elif e["event"] == "retrieval_completed":
                SAMPLES["retrieval.subquery_ms"].append(e["latency_ms"])
            elif e["event"] == "output_emitted":
                for k, v in (e.get("timings") or {}).items():
                    SAMPLES[f"engine.{k}"].append(v)
                spec = e.get("speculation") or {}
                SPEC["launched"] += spec.get("launched", 0)
                SPEC["reused"] += spec.get("reused", 0)
                post = (e.get("timings") or {}).get("post_speech_ms", 0.0)
                uid = e["utterance_id"]
                if post > 5.0 and uid in ends:   # every turn over 5 ms: what was it waiting for?
                    rel = lambda ev: round(ev["ts_ms"] - ends[uid], 1)
                    TAIL.append({"scenario": p.stem, "utterance": uid, "post_speech_ms": round(post, 2),
                                 "timings": e.get("timings"),
                                 "retrievals": [[ev["event"].split("_")[1], rel(ev), ev.get("trigger", ""),
                                                 ev.get("queries") or ev.get("query_ids")]
                                                for ev in trace if ev.get("utterance_id") == uid and ev["event"] in
                                                ("sub_queries_emitted", "retrieval_completed", "retrieval_cancelled")]})
    RECORDING.clear()
    return {"scenarios": scenarios, "corpus_dir": corpus_dir, "overrides": overrides, "time_scale": time_scale,
            "n_scenarios": len(paths), "cold_start_s": round(cold, 1),
            "execution_providers": {Path(k).parent.name + "/" + Path(k).name: v
                                    for k, v in neural.ACTIVE_PROVIDERS.items()},
            "stages": {k: summarise(v) for k, v in sorted(SAMPLES.items()) if v},
            "slowest": {k: [[round(ms, 1), sc, at] for ms, sc, at in v] for k, v in sorted(WORST.items())},
            "speculation": dict(SPEC),
            "turns_over_5ms": sorted(TAIL, key=lambda t: -t["post_speech_ms"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--out")
    ap.add_argument("--serial-gate", action="store_true",
                    help="A/B control: run the gate's reader inline (serially), as before the concurrency change")
    args = ap.parse_args(argv)
    install()
    if args.serial_gate:
        from concurrent.futures import Future

        class _Inline:
            def submit(self, fn, *a, **kw):
                f = Future()
                f.set_result(fn(*a, **kw))
                return f
        synth_mod._POOL = _Inline()
    report = asyncio.run(run(args.scenarios, args.corpus_dir, args.overrides, args.time_scale))
    print(f"{'stage':32s} {'n':>6s} {'p50':>9s} {'p95':>9s} {'p99':>9s} {'max':>9s}  (ms)")
    for k, s in report["stages"].items():
        print(f"{k:32s} {s['n']:6d} {s['p50']:9.3f} {s['p95']:9.3f} {s['p99']:9.3f} {s['max']:9.3f}")
    print(f"cold start (first scenario, excluded): {report['cold_start_s']} s")
    print(f"execution providers: {report['execution_providers']}")
    print(f"speculative final-pass retrievals: {report['speculation']}; "
          f"turns with post-speech > 5 ms: {len(report['turns_over_5ms'])}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
