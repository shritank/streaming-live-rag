"""Speculative overlap and stale-cancellation tests (Task 4 §4.2)."""
from __future__ import annotations

import asyncio
import time

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import RetrievalResult
from streaming_rag.mocks import MockController, MockRetriever, MockSynthesizer


class SlowRetriever(MockRetriever):
    """Takes a fixed 700ms per search call, like a real backend round trip."""
    def __init__(self, latency_ms=700.0, **kwargs):
        super().__init__(latency_ms=latency_ms, **kwargs)


async def test_speculative_overlap_hides_retrieval_latency(timed_event_runner):
    """Task 4 §4.2: with a mock retriever that takes 700ms, retrieval triggered
    at t=800ms completes (at t=1500ms) before utterance_end at t=2100ms — so
    turn_result must be emitted close to utterance_end (within a small
    synthesis epsilon), not at utterance_end + retrieval + synthesis."""
    events = [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "I need to plan a customer workshop"}},
        {"timestamp_ms": 800, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": " in Pune for thirty people"}},
        {"timestamp_ms": 2100, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 2200, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]
    retriever = SlowRetriever(latency_ms=700.0)

    start = time.monotonic()
    results, trace = await timed_event_runner(events, retriever=retriever, time_scale=1.0)
    first_output = next(e for e in trace if e.get("event") == "output_emitted")
    turn_wall_elapsed_s = first_output["wall_ms"] / 1000

    assert len(results) == 1
    # Real time from process start to the turn's own output_emitted: 2.1s of
    # utterance + a small synthesis tail. If the engine had instead waited
    # for utterance_end before starting retrieval, this would be
    # 2.1s + 0.7s (retrieval) = ~2.8s.
    assert turn_wall_elapsed_s < 2.5, f"took {turn_wall_elapsed_s:.2f}s — retrieval did not overlap with speech"

    retrieval_started = [e for e in trace if e.get("event") == "retrieval_started"]
    assert retrieval_started, "no retrieval was triggered before utterance_end"
    assert retrieval_started[0]["ts_ms"] < 2100, "retrieval started after utterance_end, not speculatively"


async def test_stale_retrieval_is_cancelled_and_excluded(event_runner):
    """A sub-query whose parent_query_id links it to an in-flight one must
    cancel that in-flight task, and the cancelled result must never reach
    the synthesizer."""
    from streaming_rag.contracts import ControllerDecision, Decision, SubQuery, TurnKind

    class ScriptedController(MockController):
        def __init__(self):
            self._call = 0

        def reset_utterance(self, utterance_id):
            pass

        async def on_chunk(self, chunk, session):
            self._call += 1
            if self._call == 1:
                return ControllerDecision(
                    Decision.RETRIEVE, TurnKind.NEW_REQUEST, "provisional", 0.6,
                    sub_queries=(SubQuery("u1.q1", "flights to Boston", "flights", "provisional", "u1"),),
                )
            if self._call == 2:
                return ControllerDecision(
                    Decision.RETRIEVE, TurnKind.NEW_REQUEST, "superseded", 0.9,
                    sub_queries=(SubQuery("u1.q2", "flights to New York", "flights", "final", "u1",
                                           parent_query_id="u1.q1"),),
                )
            return ControllerDecision(Decision.WAIT, TurnKind.NEW_REQUEST, "no_new_content", 1.0)

    class NeverFinishingRetriever(MockRetriever):
        async def search(self, queries, k=5):
            for q in queries:
                if q.query_id == "u1.q1":
                    await asyncio.sleep(3600)  # never resolves unless cancelled
            return await super().search(queries, k)

    class CapturingSynthesizer(MockSynthesizer):
        def __init__(self):
            super().__init__()
            self.seen_query_ids: list[str] = []

        async def synthesize(self, session_id, utterance, sub_queries, results):
            self.seen_query_ids += [r.query_id for r in results]
            return await super().synthesize(session_id, utterance, sub_queries, results)

    events = [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "flights to Boston"}},
        {"timestamp_ms": 500, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": " actually New York"}},
        {"timestamp_ms": 1000, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 5000, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]

    synthesizer = CapturingSynthesizer()
    results, trace = await event_runner(events, controller=ScriptedController(),
                                         retriever=NeverFinishingRetriever(), synthesizer=synthesizer)

    cancelled = [e for e in trace if e.get("event") == "retrieval_cancelled"]
    assert any("u1.q1" in e.get("query_ids", []) for e in cancelled), "stale query was never cancelled"
    assert "u1.q1" not in synthesizer.seen_query_ids, "cancelled result reached the synthesizer"
