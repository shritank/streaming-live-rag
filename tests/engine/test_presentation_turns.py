"""End-to-end: a reformat turn must keep the session's answer, cite nothing new
and issue no retrieval - also when there is no answer yet."""
from __future__ import annotations

from streaming_rag.config import load_config
from streaming_rag.controller import RetrievalController
from streaming_rag.mocks import MockRetriever



class CountingRetriever(MockRetriever):
    def __init__(self):
        super().__init__(latency_ms=20.0)
        self.texts: list[str] = []

    async def search(self, queries, k=5):
        self.texts += [q.text for q in queries]
        return await super().search(queries, k)


def events(*utterances):
    ev, t = [{"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}}], 0
    for i, text in enumerate(utterances, 1):
        ev.append({"timestamp_ms": t, "event_type": "transcript_chunk", "payload": {"utterance_id": f"u{i}", "text": text}})
        ev.append({"timestamp_ms": t + 1500, "event_type": "utterance_end", "payload": {"utterance_id": f"u{i}"}})
        t += 3000
    ev.append({"timestamp_ms": t + 1000, "event_type": "session_end", "payload": {}})
    return ev


async def _run(runner, *utterances):
    config = load_config()
    retriever = CountingRetriever()
    results, trace = await runner(events(*utterances), controller=RetrievalController(None, None, config),
                                  retriever=retriever, config=config, time_scale=8.0)
    return results, trace, retriever


async def test_reformat_turn_keeps_the_answer_and_never_searches(timed_event_runner):
    results, trace, retriever = await _run(timed_event_runner, "What is the catering policy for the Pune workshop?",
                                           "Now give that in one short sentence.")
    first, second = results
    assert first["retrieval_required"] is True
    assert second["retrieval_required"] is False and second["reason"] == "presentation_restructure"
    assert second["citations"] == first["citations"] and second["answer_version"] == first["answer_version"] + 1
    assert second["parent_version"] == first["answer_version"]
    assert len(retriever.texts) == len(set(retriever.texts)) and all(
        "sentence" not in t for t in retriever.texts), "the reformat request must not be searched"
    assert [e for e in trace if e["event"] == "error"] == []


async def test_reformat_without_any_answer_says_so_and_searches_nothing(timed_event_runner):
    results, trace, retriever = await _run(timed_event_runner, "Please repeat your last answer in two bullets.")
    (only,) = results
    assert only["retrieval_required"] is False and only["reason"] == "presentation_without_answer"
    assert only["citations"] == [] and "no earlier answer" in only["answer"].lower()
    assert retriever.texts == []
    assert [e for e in trace if e["event"] == "error"] == []
