"""python -m streaming_rag.cli replay <scenario.json> [--time-scale 8] [--json out.json] [--mode streaming|baseline]
python -m streaming_rag.cli live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from .build import build_components
from .config import load_config
from .contracts import NullTelemetry
from .engine import Engine
from .session.store import SessionStore
from .telemetry.sinks import JsonlTelemetry, BufferedTelemetry


def strip_ground_truth(obj):
    if isinstance(obj, dict):
        return {k: strip_ground_truth(v) for k, v in obj.items()
                 if k != "ground_truth" and not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_ground_truth(v) for v in obj]
    return obj


async def load_scenario_events(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        scenario = json.load(f)
    events = strip_ground_truth(scenario)["events"]
    return events


async def _feed_events(events: list[dict], input_queue: "asyncio.Queue", time_scale: float) -> None:
    last_ts = 0
    for e in events:
        ts = e["timestamp_ms"]
        delay = max(0, ts - last_ts) / 1000 / time_scale
        if delay:
            await asyncio.sleep(delay)
        last_ts = ts
        await input_queue.put(e)
    await input_queue.put(None)


async def _drain_output(output_queue: "asyncio.Queue", results: list[dict]) -> None:
    while True:
        item = await output_queue.get()
        if item is None:
            break
        results.append(item)


async def run_replay(scenario_path: str, time_scale: float, mode: str, telemetry_path: str | None,
                      impl: str = "real", corpus_dir: str | None = None) -> list[dict]:
    events = await load_scenario_events(scenario_path)
    config = load_config()
    config.engine.mode = mode
    config.time_scale = time_scale
    if corpus_dir:
        config.corpus_dir = corpus_dir

    telemetry = JsonlTelemetry(telemetry_path) if telemetry_path else BufferedTelemetry()
    if isinstance(telemetry, JsonlTelemetry):
        await telemetry.start()

    session_store = SessionStore()
    controller, retriever, synthesizer, llm, session_store = build_components(
        config, telemetry, session_store, impl=impl)
    engine = Engine(controller, retriever, synthesizer, llm, config, telemetry=telemetry,
                     session_store=session_store)

    input_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()
    results: list[dict] = []

    consumer = asyncio.create_task(_drain_output(output_queue, results))
    producer = asyncio.create_task(_feed_events(events, input_queue, time_scale))

    async def run_and_close():
        await engine.run(input_queue, output_queue)
        await output_queue.put(None)

    await asyncio.gather(producer, run_and_close())
    await consumer

    if isinstance(telemetry, JsonlTelemetry):
        await telemetry.stop()
    return results


GUIDE_FIELDS = ("retrieval_events", "sub_queries", "answer", "citations", "uncertainty")


def guide_record(turn_result: dict) -> dict:
    """The Theme 4 guide's structured output event record: the five fields of a
    turn_result, without the session/version bookkeeping."""
    return {k: turn_result.get(k) for k in GUIDE_FIELDS}


def print_guide_records(results: list[dict]) -> None:
    """One indented JSON record per completed turn (text and audio input alike)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # a legacy console code page must not crash on a symbol
    for r in results:
        if r.get("kind") == "turn_result":
            print(json.dumps(guide_record(r), indent=2, ensure_ascii=False, default=str))


def cmd_replay(args) -> int:
    if args.warmup:
        # Same as the evaluation harness: one uncounted, fast replay so the first CUDA calls of the process
        # (kernel loading, allocator growth) are not charged to the turn being shown. Output is discarded.
        asyncio.run(run_replay(args.scenario, 50.0, args.mode, None, impl=args.impl, corpus_dir=args.corpus_dir))
    results = asyncio.run(run_replay(args.scenario, args.time_scale, args.mode, args.telemetry,
                                       impl=args.impl, corpus_dir=args.corpus_dir))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    if args.guide_format:
        print_guide_records(results)
        return 0
    for r in results:
        print(json.dumps(r, default=str))
    return 0


async def _live_loop() -> None:
    print("Live mode: type an utterance and press Enter (blank line = utterance_end). Ctrl+C to quit.")
    config = load_config()
    telemetry = BufferedTelemetry()
    controller, retriever, synthesizer, llm, session_store = build_components(config, telemetry)
    engine = Engine(controller, retriever, synthesizer, llm, config, telemetry=telemetry,
                     session_store=session_store)
    input_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()

    async def consumer():
        while True:
            item = await output_queue.get()
            if item is None:
                break
            print(json.dumps(item, indent=2, default=str))

    engine_task = asyncio.create_task(engine.run(input_queue, output_queue))
    consumer_task = asyncio.create_task(consumer())

    session_id = "live"
    utterance_id = "u1"
    await input_queue.put({"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": session_id}})
    ts = 0
    try:
        while True:
            line = await asyncio.to_thread(input, "> ")
            ts += 300
            if line.strip() == "":
                await input_queue.put({"timestamp_ms": ts, "event_type": "utterance_end",
                                        "payload": {"utterance_id": utterance_id, "session_id": session_id}})
                utterance_id = f"u{int(utterance_id[1:]) + 1}"
            else:
                await input_queue.put({"timestamp_ms": ts, "event_type": "transcript_chunk",
                                        "payload": {"utterance_id": utterance_id, "session_id": session_id, "text": line}})
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await input_queue.put({"timestamp_ms": ts + 1, "event_type": "session_end",
                                "payload": {"session_id": session_id}})
        await input_queue.put(None)
        await engine_task
        await output_queue.put(None)
        await consumer_task


async def _show_evidence(query: str, k: int, corpus_dir: str | None) -> None:
    from .contracts import SubQuery
    config = load_config()
    if corpus_dir:
        config.corpus_dir = corpus_dir
    _, retriever, _, _, _ = build_components(config, BufferedTelemetry(), SessionStore(), impl="real")
    await retriever.setup()
    sq = SubQuery(query_id="q1", text=query, intent_label=query, trigger="final", utterance_id="u1")
    await retriever.search([sq], k=k)                       # uncounted warm-up call
    result = (await retriever.search([sq], k=k))[0]
    print(f"query: {query!r}   retrieval {result.latency_ms:.1f} ms   low_confidence={result.low_confidence}")
    print("final  citation      score   BM25-rank  dense-rank  text   (score = cross-encoder logit; ranks: 0 = best)")
    for i, e in enumerate(result.evidence, 1):
        sparse = "-" if e.sparse_rank is None else str(e.sparse_rank)
        dense = "-" if e.dense_rank is None else str(e.dense_rank)
        print(f"{i:>5}  {e.chunk.citation:<12} {e.score:>6.2f}  {sparse:>9}  {dense:>10}  {e.chunk.text[:90]!r}")


def cmd_evidence(args) -> int:
    asyncio.run(_show_evidence(args.query, args.k, args.corpus_dir))
    return 0


def cmd_live(args) -> int:
    asyncio.run(_live_loop())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="streaming_rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_replay = sub.add_parser("replay")
    p_replay.add_argument("scenario")
    p_replay.add_argument("--time-scale", type=float, default=1.0)
    p_replay.add_argument("--json", default=None)
    p_replay.add_argument("--mode", choices=["streaming", "baseline"], default="streaming")
    p_replay.add_argument("--telemetry", default=None, help="path to write a JSONL trace")
    p_replay.add_argument("--impl", choices=["real", "mock"], default="real")
    p_replay.add_argument("--corpus-dir", default=None)
    p_replay.add_argument("--warmup", action="store_true",
                          help="run one uncounted fast replay first, so cold-start GPU costs are not shown as latency")
    p_replay.add_argument("--guide-format", action="store_true",
                          help="print each turn as the guide's structured output event record "
                               "(retrieval_events, sub_queries, answer, citations, uncertainty)")
    p_replay.set_defaults(func=cmd_replay)

    p_evidence = sub.add_parser("evidence", help="show one query's fused + reranked evidence with per-channel ranks")
    p_evidence.add_argument("query")
    p_evidence.add_argument("--k", type=int, default=6)
    p_evidence.add_argument("--corpus-dir", default=None)
    p_evidence.set_defaults(func=cmd_evidence)

    p_live = sub.add_parser("live")
    p_live.set_defaults(func=cmd_live)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
