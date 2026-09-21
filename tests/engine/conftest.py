from __future__ import annotations

import asyncio

import pytest

from streaming_rag.config import load_config
from streaming_rag.engine import Engine
from streaming_rag.mocks import MockController, MockRetriever, MockSynthesizer
from streaming_rag.session.store import SessionStore
from streaming_rag.telemetry.sinks import BufferedTelemetry


async def run_events(events: list[dict], mode: str = "streaming",
                      controller=None, retriever=None, synthesizer=None,
                      config=None) -> tuple[list[dict], list[dict]]:
    config = config or load_config()
    config.engine.mode = mode
    telemetry = BufferedTelemetry()
    controller = controller or MockController()
    retriever = retriever or MockRetriever(latency_ms=5.0)
    synthesizer = synthesizer or MockSynthesizer()
    engine = Engine(controller, retriever, synthesizer, None, config, telemetry=telemetry,
                     session_store=SessionStore())

    input_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()
    results: list[dict] = []

    async def drain():
        while True:
            item = await output_queue.get()
            if item is None:
                break
            results.append(item)

    consumer = asyncio.create_task(drain())

    async def feed():
        for e in events:
            await input_queue.put(e)
        await input_queue.put(None)

    async def run_and_close():
        await engine.run(input_queue, output_queue)
        await output_queue.put(None)

    await asyncio.gather(feed(), run_and_close())
    await consumer
    return results, telemetry.events


async def run_events_timed(events: list[dict], mode: str = "streaming",
                            controller=None, retriever=None, synthesizer=None,
                            config=None, time_scale: float = 1.0) -> tuple[list[dict], list[dict]]:
    """Like run_events, but respects each event's timestamp_ms gap in real
    wall-clock time (scaled), so tests can assert on actual overlap/latency."""
    config = config or load_config()
    config.engine.mode = mode
    config.time_scale = time_scale
    telemetry = BufferedTelemetry()
    controller = controller or MockController()
    retriever = retriever or MockRetriever(latency_ms=5.0)
    synthesizer = synthesizer or MockSynthesizer()
    engine = Engine(controller, retriever, synthesizer, None, config, telemetry=telemetry,
                     session_store=SessionStore())

    input_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()
    results: list[dict] = []

    async def drain():
        while True:
            item = await output_queue.get()
            if item is None:
                break
            results.append(item)

    consumer = asyncio.create_task(drain())

    async def feed():
        last_ts = 0
        for e in events:
            ts = e["timestamp_ms"]
            delay = max(0, ts - last_ts) / 1000 / time_scale
            if delay:
                await asyncio.sleep(delay)
            last_ts = ts
            await input_queue.put(e)
        await input_queue.put(None)

    async def run_and_close():
        await engine.run(input_queue, output_queue)
        await output_queue.put(None)

    await asyncio.gather(feed(), run_and_close())
    await consumer
    return results, telemetry.events


@pytest.fixture
def event_runner():
    return run_events


@pytest.fixture
def timed_event_runner():
    return run_events_timed
