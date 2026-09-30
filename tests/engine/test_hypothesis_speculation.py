"""transcript_hypothesis events (the streaming ASR's uncommitted words) may
only warm the speculative cache: outputs must be identical with or without
them, right or wrong, and a correct hypothesis must actually be reused."""
from __future__ import annotations

from streaming_rag.config import load_config
from streaming_rag.controller import RetrievalController
from streaming_rag.engine import Engine
from streaming_rag.mocks import MockRetriever


def events(hypothesis: str | None):
    ev = [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "What is the cancellation"}},
    ]
    if hypothesis is not None:   # ASR's full hypothesis, ~0.4 s before its final flush
        ev.append({"timestamp_ms": 600, "event_type": "transcript_hypothesis",
                   "payload": {"session_id": "s1", "utterance_id": "u1", "text": hypothesis}})
    ev += [
        # final flush: the rest of the question and utterance_end 1 ms apart (audio path)
        {"timestamp_ms": 1000, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": " policy for the Pune workshop?"}},
        {"timestamp_ms": 1001, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 2000, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]
    return ev


async def _run(runner, hypothesis):
    config = load_config()
    return await runner(events(hypothesis), controller=RetrievalController(None, None, config),
                        retriever=MockRetriever(latency_ms=30.0), config=config, time_scale=2.0)


def _out(results):
    return [(r["answer"], r["citations"], r["uncertainty"], r["sub_queries"]) for r in results]


def _spec(trace):
    return [e["speculation"] for e in trace if e["event"] == "output_emitted"][0]


async def test_correct_hypothesis_is_reused_and_changes_nothing(timed_event_runner):
    base, _ = await _run(timed_event_runner, None)
    hit, trace = await _run(timed_event_runner, "What is the cancellation policy for the Pune workshop?")
    assert _out(hit) == _out(base)
    assert _spec(trace)["reused"] >= 1


async def test_wrong_hypothesis_changes_nothing(timed_event_runner):
    base, _ = await _run(timed_event_runner, None)
    miss, trace = await _run(timed_event_runner, "What is the cancellation fee for catering?")
    assert _out(miss) == _out(base)
    assert _spec(trace)["reused"] == 0


def test_hypothesis_contradicting_committed_words_is_ignored():
    assert Engine._pending_from_hypothesis("What is the cancellation", "What was the cancellation policy") == ""
    assert Engine._pending_from_hypothesis("What is the", "What is the") == ""
    assert Engine._pending_from_hypothesis("What is the", "What is the policy?") == " policy?"
    assert Engine._pending_from_hypothesis("", "What is") == "What is"
