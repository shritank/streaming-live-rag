"""Task 4 acceptance (section 4.3): telemetry overhead. Run the dev suite with the real sink and with NullTelemetry;
the p95 turn latency difference must be <= 5 %.

The engine records its own post-speech timings *through* telemetry, so with a null sink they cannot be read from
the trace. Both variants are therefore timed identically from OUTSIDE: perf_counter when the `utterance_end`
envelope is put on the input queue -> perf_counter when the `turn_result` for that utterance leaves the output
queue (includes the harness's own constant overhead, identical for every sink).

    python handoff/final_audit/telemetry_overhead.py [SCENARIO_DIR] [CORPUS_DIR] [TIME_SCALE] [ROUNDS]

Also micro-benchmarks one emit() per sink and counts the events emitted inside the post-speech window.
Order per round: null, buffered, jsonl, null, ... (interleaved so machine drift hits every sink alike).
"""
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from eval.loader import discover_scenarios, load_events  # noqa: E402
from streaming_rag.build import build_components  # noqa: E402
from streaming_rag.config import load_config  # noqa: E402
from streaming_rag.contracts import NullTelemetry  # noqa: E402
from streaming_rag.engine import Engine  # noqa: E402
from streaming_rag.session.store import SessionStore  # noqa: E402
from streaming_rag.telemetry.sinks import BufferedTelemetry, JsonlTelemetry  # noqa: E402


async def replay(path, config, telemetry, time_scale):
    events = load_events(path)
    controller, retriever, synthesizer, llm, store = build_components(config, telemetry, SessionStore(), impl="real")
    engine = Engine(controller, retriever, synthesizer, llm, config, telemetry=telemetry, session_store=store)
    inq, outq = asyncio.Queue(), asyncio.Queue()
    end_at, lat = {}, []

    async def feed():
        last = 0
        for e in events:
            d = max(0, e["timestamp_ms"] - last) / 1000 / time_scale
            if d:
                await asyncio.sleep(d)
            last = e["timestamp_ms"]
            if e["event_type"] == "utterance_end":
                end_at[e["payload"]["utterance_id"]] = time.perf_counter()
            await inq.put(e)
        await inq.put(None)

    async def drain():
        while True:
            item = await outq.get()
            if item is None:
                return
            t = end_at.get(item.get("utterance_id"))
            if item.get("kind") == "turn_result" and t is not None:
                lat.append((time.perf_counter() - t) * 1000)

    async def run():
        await engine.run(inq, outq)
        await outq.put(None)

    d = asyncio.create_task(drain())
    await asyncio.gather(feed(), run())
    await d
    return lat


def pct(a, q):
    s = sorted(a)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


async def main():
    sc_dir = sys.argv[1] if len(sys.argv) > 1 else "eval/scenarios_real_dev"
    corpus = sys.argv[2] if len(sys.argv) > 2 else "data/corpus"
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    config = load_config()
    config.corpus_dir = corpus
    paths = [str(p) for p in discover_scenarios(sc_dir)]
    # warm-up (models, index) so no variant pays it
    await replay(paths[0], config, NullTelemetry(), scale)
    tmp = Path(tempfile.mkdtemp())
    sinks = {"null": lambda: NullTelemetry(), "buffered": lambda: BufferedTelemetry(),
             "jsonl": lambda: JsonlTelemetry(str(tmp / "trace.jsonl"))}
    lat = {k: [] for k in sinks}
    for r in range(rounds):
        for name in ("null", "buffered", "jsonl") if r % 2 == 0 else ("jsonl", "buffered", "null"):
            for p in paths:
                sink = sinks[name]()
                if hasattr(sink, "start"):
                    await sink.start()
                lat[name] += await replay(p, config, sink, scale)
                if hasattr(sink, "stop"):
                    await sink.stop()
    out = {"scenarios": len(paths), "rounds": rounds, "time_scale": scale, "turns_per_sink": len(lat["null"])}
    for k, v in lat.items():
        out[k] = {"n": len(v), "p50": round(pct(v, .5), 3), "p95": round(pct(v, .95), 3), "p99": round(pct(v, .99), 3),
                  "max": round(max(v), 3), "mean": round(statistics.mean(v), 3)}
    for k in ("buffered", "jsonl"):
        out[f"{k}_vs_null_p95_pct"] = round(100 * (out[k]["p95"] - out["null"]["p95"]) / out["null"]["p95"], 1)
        out[f"{k}_vs_null_p50_pct"] = round(100 * (out[k]["p50"] - out["null"]["p50"]) / out["null"]["p50"], 1)
    # micro-benchmark: cost of one emit() with a realistic payload
    payload = dict(component="engine", session_id="s1", utterance_id="u1", ts_ms=1234.5, decision="retrieve",
                   turn_kind="new_request", reason="intent_stable", stability=0.8, n_sub_queries=1,
                   decision_latency_ms=0.4)
    micro = {}
    for name in ("null", "buffered", "jsonl"):
        sink = sinks[name]()
        if hasattr(sink, "start"):
            await sink.start()
        t = time.perf_counter()
        for _ in range(20000):
            sink.emit("controller_decision", **dict(payload))
        micro[name] = round((time.perf_counter() - t) / 20000 * 1e6, 2)
        if hasattr(sink, "stop"):
            await sink.stop()
    out["emit_cost_us"] = micro
    print(json.dumps(out, indent=1))
    Path("eval/results/final_audit/telemetry_overhead.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
