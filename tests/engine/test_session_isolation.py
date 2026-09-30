"""Session-bound state (hard rule): two sessions that happen to use the SAME utterance id ("u1" is the guide's own
example id) and speak at the same time must not see each other's words, sub-queries or answers."""
from __future__ import annotations

from streaming_rag.config import load_config
from streaming_rag.controller import RetrievalController
from streaming_rag.mocks import MockRetriever


def ev(t, typ, **payload):
    return {"timestamp_ms": t, "event_type": typ, "payload": payload}


def two_sessions_same_utterance_id():
    return [
        ev(0, "session_start", session_id="A"), ev(0, "session_start", session_id="B"),
        ev(0, "transcript_chunk", session_id="A", utterance_id="u1", text="What is the catering policy"),
        ev(100, "transcript_chunk", session_id="B", utterance_id="u1", text="How long is the cancellation window"),
        ev(700, "transcript_chunk", session_id="A", utterance_id="u1", text=" for the Pune workshop?"),
        ev(800, "transcript_chunk", session_id="B", utterance_id="u1", text=" for a venue booking?"),
        ev(1500, "utterance_end", session_id="A", utterance_id="u1"),
        ev(1600, "utterance_end", session_id="B", utterance_id="u1"),
        ev(3000, "session_end", session_id="A"), ev(3000, "session_end", session_id="B"),
    ]


async def test_same_utterance_id_in_two_live_sessions_does_not_mix_state(timed_event_runner):
    config = load_config()
    results, trace = await timed_event_runner(two_sessions_same_utterance_id(),
                                              controller=RetrievalController(None, None, config),
                                              retriever=MockRetriever(latency_ms=20.0), config=config, time_scale=4.0)
    by_session = {r["session_id"]: r for r in results}
    assert set(by_session) == {"A", "B"}, [r["session_id"] for r in results]
    a_text = " ".join(by_session["A"]["sub_queries"]).lower()
    b_text = " ".join(by_session["B"]["sub_queries"]).lower()
    assert "catering" in a_text and "cancellation" not in a_text, by_session["A"]["sub_queries"]
    assert "cancellation" in b_text and "catering" not in b_text, by_session["B"]["sub_queries"]


def one_session_reusing_an_utterance_id():
    return [
        ev(0, "session_start", session_id="A"),
        ev(0, "transcript_chunk", session_id="A", utterance_id="u1", text="What is the catering policy for the Pune workshop?"),
        ev(1000, "utterance_end", session_id="A", utterance_id="u1"),
        ev(2500, "transcript_chunk", session_id="A", utterance_id="u1", text="How long is the cancellation window for a venue booking?"),
        ev(3500, "utterance_end", session_id="A", utterance_id="u1"),
        ev(5000, "session_end", session_id="A"),
    ]


async def test_a_reused_utterance_id_starts_a_fresh_utterance(timed_event_runner):
    """Regression: the controller kept the old utterance's text forever, so a later turn that reused the id had the
    earlier question glued onto it (and its sub-queries suppressed as 'already covered')."""
    config = load_config()
    results, _ = await timed_event_runner(one_session_reusing_an_utterance_id(),
                                          controller=RetrievalController(None, None, config),
                                          retriever=MockRetriever(latency_ms=20.0), config=config, time_scale=4.0)
    assert len(results) == 2
    second = " ".join(results[1]["sub_queries"]).lower()
    assert "cancellation" in second and "catering" not in second, results[1]["sub_queries"]


def _controller_keys(c):
    return [k for d in (c._accumulated, c._previous, c._covered, c._issued, c._live, c._clauses) for k in d]


async def test_the_controller_keeps_no_state_once_the_turn_and_the_session_are_over(timed_event_runner):
    """Session memory is ephemeral: nothing of a finished utterance / session may stay in the controller. The
    session_end below has an EMPTY payload, exactly as in the guide's input example."""
    config = load_config()
    controller = RetrievalController(None, None, config)
    events = two_sessions_same_utterance_id()[:-2] + [ev(3000, "session_end")]
    events = [e for e in events if not (e["event_type"] == "session_end" and e["payload"].get("session_id") == "B")]
    events = [e for e in events if e["payload"].get("session_id") != "B"]        # one session, guide-style end
    await timed_event_runner(events, controller=controller, retriever=MockRetriever(latency_ms=20.0),
                             config=config, time_scale=4.0)
    assert _controller_keys(controller) == []
