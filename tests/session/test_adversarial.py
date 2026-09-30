"""Adversarial and runtime-contract tests, ported from build A's suite.

Build C tested answer *quality* thoroughly but not these properties, which the
project rules require and which no gate measures:

* the event loop is never blocked (project context §9.2: a blocked loop delays
  every event behind it and corrupts every latency number);
* sessions never leak into each other, and close() really forgets;
* session state is never written to disk (hard rule: session-bound state);
* a presentation-only turn with nothing to present degrades instead of crashing.
"""
from __future__ import annotations

import asyncio
import builtins
import pathlib
import time

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import Chunk, EvidenceChunk, RetrievalResult, SubQuery
from streaming_rag.retrieval import HybridRetriever
from streaming_rag.session.store import SessionStore
from streaming_rag.session.synthesis import GroundedSynthesizer

CHUNKS = [
    Chunk(chunk_id="Doc_01 §1 #1", doc_id="Doc_01", section="1",
          text="International travel requires senior director approval before booking."),
    Chunk(chunk_id="Doc_01 §2 #1", doc_id="Doc_01", section="2",
          text="Standard reimbursement covers economy airfare and lodging."),
    Chunk(chunk_id="Doc_02 §1 #1", doc_id="Doc_02", section="1",
          text="The Aurora hall seats thirty attendees and is available on weekdays."),
]


def _sq(query_id: str, text: str, utterance_id: str = "u1") -> SubQuery:
    return SubQuery(query_id=query_id, text=text, intent_label=text[:24],
                    trigger="final", utterance_id=utterance_id)


def _result(query_id: str, chunk: Chunk, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(query_id=query_id,
                           evidence=(EvidenceChunk(chunk=chunk, score=score, query_ids=(query_id,)),),
                           latency_ms=1.0, low_confidence=False)


async def _build(selector: str | None = None):
    cfg = load_config()
    if selector is not None:
        cfg.synthesis.selector = selector
    retriever = HybridRetriever(cfg, chunks=CHUNKS)
    await retriever.setup()
    store = SessionStore()
    return GroundedSynthesizer(retriever, store, cfg), store


async def _max_loop_gap(coro) -> float:
    """Run `coro` while a 10 ms ticker measures the worst event-loop stall."""
    gaps: list[float] = []
    done = asyncio.Event()

    async def ticker() -> None:
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    # Let the ticker start sleeping first. A coroutine that never yields would
    # otherwise run to completion before the ticker's first turn, and the
    # stall would be invisible (the measurement would report 0).
    await asyncio.sleep(0.02)
    try:
        await coro
        await asyncio.sleep(0.02)   # let the ticker observe the late wake-up
    finally:
        done.set()
        await tick
    return max(gaps) if gaps else 0.0


# ---------------------------------------------------------------- event loop
@pytest.mark.asyncio
async def test_neural_selector_does_not_block_the_event_loop():
    """A prefetch miss must not run cross-encoder + reader inference on the loop.

    The engine prefetches claims during speech, but a refinement or a
    retrieval that lands at utterance end misses that cache, and the model
    inference then runs inside synthesize(). On the loop it would freeze every
    other coroutine for the full inference time.
    """
    synth, store = await _build(selector="cross-encoder")
    store.open("s1")
    # warm the models so the measurement is inference, not one-off loading
    await synth.synthesize("s1", "warm", [_sq("w", "who approves international travel")],
                           [_result("w", CHUNKS[0])])

    texts = ["who must approve international travel bookings",
             "what does standard reimbursement cover",
             "how many attendees does the Aurora hall seat"]
    # Six sub-queries: the stall grows ~25 ms per sub-query on a fast CPU, so
    # three was small enough to hide a blocked loop under a loose threshold.
    queries = [_sq(f"q{i}", texts[i % 3]) for i in range(6)]
    results = [_result(q.query_id, CHUNKS[i % 3]) for i, q in enumerate(queries)]

    gap = await _max_loop_gap(synth.synthesize("s1", "six questions", queries, results))
    assert gap < 0.05, f"event loop stalled for {gap * 1000:.0f} ms during synthesis"


# ---------------------------------------------------------------- sessions
@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_leak_answers():
    synth, store = await _build()
    store.open("a")
    store.open("b")

    va, vb = await asyncio.gather(
        synth.synthesize("a", "travel", [_sq("a1", "who approves international travel")],
                         [_result("a1", CHUNKS[0])]),
        synth.synthesize("b", "hall", [_sq("b1", "how many attendees does the hall seat")],
                         [_result("b1", CHUNKS[2])]),
    )
    store.record_answer("a", va)
    store.record_answer("b", vb)

    a_cites = set(store.view("a").latest_answer().citations)
    b_cites = set(store.view("b").latest_answer().citations)
    assert "Doc_02 §1" not in a_cites
    assert "Doc_01 §1" not in b_cites


@pytest.mark.asyncio
async def test_close_forgets_the_session():
    synth, store = await _build()
    store.open("s1")
    v1 = await synth.synthesize("s1", "travel", [_sq("q1", "who approves international travel")],
                                [_result("q1", CHUNKS[0])])
    store.record_answer("s1", v1)
    assert store.view("s1").has_answer()

    store.close("s1")

    # A new conversation that reuses the id must start empty.
    assert not store.view("s1").has_answer()
    assert store.view("s1").prior_sub_queries() == []


@pytest.mark.asyncio
async def test_no_session_state_is_written_to_disk(monkeypatch):
    synth, store = await _build()     # index build may cache embeddings: allowed
    store.open("s1")
    writes: list[str] = []
    real_open = builtins.open

    def spy_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(pathlib.Path, "write_text", lambda self, *a, **k: writes.append(str(self)))
    monkeypatch.setattr(pathlib.Path, "write_bytes", lambda self, *a, **k: writes.append(str(self)))

    v1 = await synth.synthesize("s1", "travel", [_sq("q1", "who approves international travel")],
                                [_result("q1", CHUNKS[0])])
    store.record_answer("s1", v1)
    v2 = await synth.refine("s1", "and reimbursement",
                            [_sq("q2", "what does standard reimbursement cover", "u2")],
                            [_result("q2", CHUNKS[1])])
    store.record_answer("s1", v2)
    store.record_answer("s1", await synth.restructure("s1", "two bullets"))
    store.close("s1")

    assert writes == [], f"session work wrote to disk: {writes}"


# ---------------------------------------------------------------- degradation
@pytest.mark.asyncio
async def test_presentation_turn_with_no_prior_answer_does_not_crash_the_engine():
    """'Repeat that' as the very first turn has nothing to restructure.

    C's synthesizer raises in that case. The engine must still answer the turn
    (with an uncertainty note) rather than drop it or die.
    """
    from streaming_rag.contracts import ControllerDecision, Decision, TurnKind
    from tests.engine.conftest import run_events

    class PresentationOnly:
        async def on_chunk(self, chunk, session):
            if not chunk.is_final:
                return ControllerDecision(decision=Decision.WAIT, turn_kind=TurnKind.PRESENTATION_ONLY,
                                          reason="waiting", stability=0.2)
            return ControllerDecision(decision=Decision.SUPPRESS, turn_kind=TurnKind.PRESENTATION_ONLY,
                                      reason="presentation_restructure", stability=1.0)

        def reset_utterance(self, utterance_id):
            pass

    synth, _ = await _build()
    events = [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"utterance_id": "u1", "text": "Please repeat your last answer in two bullets."}},
        {"timestamp_ms": 800, "event_type": "utterance_end", "payload": {"utterance_id": "u1"}},
        {"timestamp_ms": 3000, "event_type": "session_end", "payload": {}},
    ]
    outputs, trace = await asyncio.wait_for(
        run_events(events, controller=PresentationOnly(), synthesizer=synth), timeout=30)

    turn_results = [o for o in outputs if o.get("kind") == "turn_result"]
    assert turn_results, f"the turn produced no result at all: {outputs}"
    assert turn_results[-1].get("citations") == []
    assert not any(e.get("event") == "error" and e.get("fatal") for e in trace)
