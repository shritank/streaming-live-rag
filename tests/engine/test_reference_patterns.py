"""Replays the guide's three reference patterns with scripted mocks and
asserts the turn_result shape matches §6.6 / the guide's §5 record exactly:
keys, types, timestamp_s in seconds, trigger values (Task 4 §4.2).
"""
from __future__ import annotations

import asyncio

import pytest

from streaming_rag.mocks import MockController, MockRetriever, MockSynthesizer

EXPECTED_KEYS = {
    "kind", "session_id", "utterance_id", "retrieval_events", "sub_queries",
    "answer", "citations", "uncertainty", "answer_version", "parent_version",
    "retrieval_required", "reason",
}


def multi_intent_events():
    return [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "I need to plan a customer workshop in"}},
        {"timestamp_ms": 800, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": " Pune for 30 people, and I need"}},
        {"timestamp_ms": 1600, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": " the cancellation policy and the catering options."}},
        {"timestamp_ms": 2100, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 9000, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]


async def test_example1_multi_intent_output_shape(event_runner):
    results, trace = await event_runner(multi_intent_events())
    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) == EXPECTED_KEYS
    assert isinstance(r["retrieval_events"], list)
    for ev in r["retrieval_events"]:
        assert set(ev.keys()) == {"timestamp_s", "query", "trigger"}
        assert isinstance(ev["timestamp_s"], (int, float))
        assert ev["trigger"] in ("provisional", "multi_intent", "refinement", "final")
    assert isinstance(r["citations"], list)
    assert isinstance(r["answer_version"], int)
    assert r["parent_version"] is None
    assert r["retrieval_required"] is True


def late_detail_events():
    return [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "Summarize the travel reimbursement rule for an employee trip"}},
        {"timestamp_ms": 2000, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 3000, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u2", "text": "Actually, the trip was international and the booking was made after travel."}},
        {"timestamp_ms": 5000, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u2"}},
        {"timestamp_ms": 9000, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]


async def test_example2_late_detail_refines_not_restarts(event_runner):
    results, trace = await event_runner(late_detail_events())
    assert len(results) == 2
    v1, v2 = results
    assert v1["answer_version"] == 1
    assert v1["parent_version"] is None
    assert v2["answer_version"] == 2
    assert v2["parent_version"] == 1
    # prior citations kept
    assert set(v1["citations"]) <= set(v2["citations"])
    # G5: the delta must not re-run every prior sub_query verbatim
    assert v2["sub_queries"] != v1["sub_queries"] + v1["sub_queries"]


def suppression_events():
    return [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "What is the cancellation policy"}},
        {"timestamp_ms": 1500, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 2500, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u2", "text": "Please repeat your last answer in two bullets."}},
        {"timestamp_ms": 3500, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u2"}},
        {"timestamp_ms": 9000, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]


async def test_example3_query_suppression_no_retrieval(event_runner):
    class TrackingRetriever(MockRetriever):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        async def search(self, queries, k=5):
            self.call_count += 1
            return await super().search(queries, k)

    class TrackingSynthesizer(MockSynthesizer):
        def __init__(self):
            super().__init__()
            self.synthesize_calls = 0

        async def synthesize(self, *args, **kwargs):
            self.synthesize_calls += 1
            return await super().synthesize(*args, **kwargs)

    retriever = TrackingRetriever()
    synthesizer = TrackingSynthesizer()
    results, trace = await event_runner(suppression_events(), retriever=retriever, synthesizer=synthesizer)

    assert len(results) == 2
    v1, v2 = results
    assert v2["retrieval_required"] is False
    assert v2["reason"] == "presentation_restructure"
    assert v2["retrieval_events"] == []
    # synthesize() was called exactly once (for u1); restructure() handled u2
    assert synthesizer.synthesize_calls == 1
    # no new citations were fabricated on the presentation-only turn
    assert set(v2["citations"]) <= set(v1["citations"]) or set(v2["citations"]) == set(v1["citations"])
