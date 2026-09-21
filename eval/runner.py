"""Runs one scenario through a fresh engine instance and returns both the
turn_results and the raw telemetry trace, which is all the gate scorers
need (they never touch ground_truth directly outside this module's caller).
"""
from __future__ import annotations

import asyncio

from streaming_rag.build import build_components
from streaming_rag.config import Config
from streaming_rag.engine import Engine
from streaming_rag.session.store import SessionStore
from streaming_rag.telemetry.sinks import BufferedTelemetry

from .loader import load_events


async def _feed(events: list[dict], input_queue: "asyncio.Queue", time_scale: float) -> None:
    last_ts = 0
    for e in events:
        ts = e["timestamp_ms"]
        delay = max(0, ts - last_ts) / 1000 / max(time_scale, 1e-9)
        if delay:
            await asyncio.sleep(delay)
        last_ts = ts
        await input_queue.put(e)
    await input_queue.put(None)


async def run_scenario(scenario_path: str, config: Config, impl: str = "real",
                        time_scale: float = 8.0) -> tuple[list[dict], list[dict]]:
    """Returns (turn_results, trace_events) for one replay."""
    events = load_events(scenario_path)
    telemetry = BufferedTelemetry()
    session_store = SessionStore()
    controller, retriever, synthesizer, llm, session_store = build_components(
        config, telemetry, session_store, impl=impl)
    engine = Engine(controller, retriever, synthesizer, llm, config, telemetry=telemetry,
                     session_store=session_store)

    input_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()
    results: list[dict] = []

    async def drain():
        while True:
            item = await output_queue.get()
            if item is None:
                break
            results.append(item)

    async def run_and_close():
        await engine.run(input_queue, output_queue)
        await output_queue.put(None)

    consumer = asyncio.create_task(drain())
    producer = asyncio.create_task(_feed(events, input_queue, time_scale))
    await asyncio.gather(producer, run_and_close())
    await consumer

    return results, telemetry.events


async def run_scenario_median(scenario_path: str, config: Config, impl: str = "real",
                               time_scale: float = 8.0, reps: int = 3):
    """Runs the scenario `reps` times with fresh engine instances and returns
    the median run by trace length (project context §9.8: reps & median)."""
    runs = [await run_scenario(scenario_path, config, impl, time_scale) for _ in range(reps)]
    runs.sort(key=lambda r: len(r[1]))
    return runs[len(runs) // 2]
