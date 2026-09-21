"""Robustness & fault injection (Task 4 §4.7). Each mock is made to raise,
time out, return empty, or return malformed data — the engine must never
crash and must still produce a degraded turn_result for that turn.
"""
from __future__ import annotations

import asyncio

import pytest

from streaming_rag.config import load_config
from streaming_rag.mocks import MockController, MockRetriever, MockSynthesizer

BASIC_EVENTS = [
    {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
    {"timestamp_ms": 0, "event_type": "transcript_chunk",
     "payload": {"session_id": "s1", "utterance_id": "u1", "text": "book a flight to Boston"}},
    {"timestamp_ms": 1000, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
    {"timestamp_ms": 1500, "event_type": "session_end", "payload": {"session_id": "s1"}},
]


@pytest.mark.parametrize("fail_mode", ["raise", "empty", "malformed"])
async def test_retriever_fault_injection_never_crashes(event_runner, fail_mode):
    retriever = MockRetriever(fail_mode=fail_mode)
    results, trace = await event_runner(BASIC_EVENTS, retriever=retriever)
    assert len(results) == 1
    assert results[0]["kind"] == "turn_result"
    if fail_mode in ("raise", "empty"):
        assert results[0]["uncertainty"] is not None or results[0]["citations"] == []


async def test_retriever_timeout_degrades_via_turn_timeout(event_runner):
    retriever = MockRetriever(fail_mode="timeout")
    config = load_config()
    config.engine.per_turn_timeout_ms = 200  # short cap so the test doesn't hang
    results, trace = await event_runner(BASIC_EVENTS, retriever=retriever, config=config)
    assert len(results) == 1
    assert results[0]["reason"] == "degraded"
    errors = [e for e in trace if e.get("event") == "error" and e.get("error_type") == "turn_timeout"]
    assert errors, "expected a turn_timeout error event"


@pytest.mark.parametrize("fail_mode", ["raise"])
async def test_synthesizer_fault_injection_never_crashes(event_runner, fail_mode):
    synthesizer = MockSynthesizer(fail_mode=fail_mode)
    results, trace = await event_runner(BASIC_EVENTS, synthesizer=synthesizer)
    assert len(results) == 1
    assert results[0]["reason"] == "degraded"
    errors = [e for e in trace if e.get("event") == "error" and e.get("where") == "synthesizer"]
    assert errors


async def test_controller_fault_injection_never_crashes(event_runner):
    controller = MockController(fail_mode="raise")
    results, trace = await event_runner(BASIC_EVENTS, controller=controller)
    errors = [e for e in trace if e.get("event") == "error" and e.get("where") == "controller.on_chunk"]
    assert errors
    # the session still finishes cleanly, i.e. no exception escapes run()


async def test_malformed_events_are_logged_and_skipped(event_runner):
    events = [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"event_type": "transcript_chunk", "payload": {"session_id": "s1", "utterance_id": "u1", "text": "hi"}},  # missing timestamp_ms
        {"timestamp_ms": 100, "event_type": "not_a_real_event", "payload": {}},
        {"timestamp_ms": 200, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "book a flight to Boston"}},
        {"timestamp_ms": 1000, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 1200, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]
    results, trace = await event_runner(events)
    assert len(results) == 1  # the session still completes cleanly
    error_types = {e.get("error_type") for e in trace if e.get("event") == "error"}
    assert "malformed_event" in error_types
    assert "unknown_event_type" in error_types


async def test_missing_utterance_end_before_session_end_does_not_crash(event_runner):
    events = [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "book a flight to Boston"}},
        {"timestamp_ms": 500, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]
    results, trace = await event_runner(events)
    # no crash; the utterance simply never produces a turn_result
    assert results == []


async def test_loop_lag_watchdog_detects_a_blocking_call():
    """Injects a component that blocks the loop synchronously for 300ms and
    asserts the watchdog reports loop_lag, proving the detector works."""
    import time as time_module
    from streaming_rag.engine import Engine
    from streaming_rag.session.store import SessionStore
    from streaming_rag.telemetry.sinks import BufferedTelemetry

    class BlockingController(MockController):
        async def on_chunk(self, chunk, session):
            time_module.sleep(0.3)  # a real synchronous block — the bug this watchdog exists to catch
            return await super().on_chunk(chunk, session)

    config = load_config()
    config.engine.loop_lag_threshold_ms = 50
    telemetry = BufferedTelemetry()
    engine = Engine(BlockingController(), MockRetriever(), MockSynthesizer(), None, config,
                     telemetry=telemetry, session_store=SessionStore())

    input_queue: asyncio.Queue = asyncio.Queue()
    output_queue: asyncio.Queue = asyncio.Queue()

    async def drain():
        while await output_queue.get() is not None:
            pass

    consumer = asyncio.create_task(drain())
    for e in BASIC_EVENTS:
        await input_queue.put(e)
    await input_queue.put(None)

    await engine.run(input_queue, output_queue)
    await output_queue.put(None)
    await consumer

    lag_errors = [e for e in telemetry.events if e.get("event") == "error" and e.get("error_type") == "loop_lag"]
    assert lag_errors, "loop-lag watchdog did not fire for a 300ms synchronous block"
