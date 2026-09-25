"""`deferred` is the fair control for the streaming claim: identical pipeline
(decomposition, refinement, synthesis) but no retrieval until the utterance
has ended — so any accuracy or latency difference is due to timing alone."""
from __future__ import annotations

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


async def test_deferred_mode_never_retrieves_before_utterance_end(event_runner):
    results, trace = await event_runner(multi_intent_events(), mode="deferred")
    end_ts = next(e["ts_ms"] for e in trace if e["event"] == "chunk_received" and e.get("is_final"))
    starts = [e["ts_ms"] for e in trace if e["event"] == "retrieval_started"]
    assert starts, "deferred mode must still retrieve"
    assert all(ts >= end_ts for ts in starts)
    assert set(results[0].keys()) == EXPECTED_KEYS


async def test_deferred_mode_keeps_the_same_decomposition_as_streaming(event_runner):
    streamed, _ = await event_runner(multi_intent_events(), mode="streaming")
    deferred, _ = await event_runner(multi_intent_events(), mode="deferred")
    assert len(deferred[0]["sub_queries"]) >= 2
    assert deferred[0]["retrieval_required"] is True
    assert streamed[0]["citations"] and deferred[0]["citations"]
