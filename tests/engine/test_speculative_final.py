"""Speculative final-pass retrieval (engine.speculative_final): must not change
any output, and must actually reuse the speculated retrieval."""
from __future__ import annotations

from streaming_rag.config import load_config
from streaming_rag.controller import RetrievalController
from streaming_rag.mocks import MockRetriever


def events():
    # a short, still-"unstable" question: the controller WAITs on the chunk and
    # only its final pass at utterance_end issues the sub-query (the pattern of
    # dev scenario gen_late_detail_000) - the case speculation exists for
    return [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"session_id": "s1", "utterance_id": "u1", "text": "What is the catering policy"}},
        {"timestamp_ms": 1200, "event_type": "utterance_end", "payload": {"session_id": "s1", "utterance_id": "u1"}},
        {"timestamp_ms": 2000, "event_type": "session_end", "payload": {"session_id": "s1"}},
    ]


class CountingRetriever(MockRetriever):
    def __init__(self):
        super().__init__(latency_ms=20.0)
        self.texts: list[str] = []

    async def search(self, queries, k=5):
        self.texts += [q.text for q in queries]
        return await super().search(queries, k)


async def _run(runner, speculative: bool):
    config = load_config()
    config.engine.speculative_final = speculative
    retriever = CountingRetriever()
    results, trace = await runner(events(), controller=RetrievalController(None, None, config),
                                            retriever=retriever, config=config, time_scale=4.0)
    return results, trace, retriever


def _comparable(results):
    return [(r["answer"], r["citations"], r["uncertainty"], [q for q in r["sub_queries"]]) for r in results]


async def test_speculation_changes_no_output_and_is_reused(timed_event_runner):
    off, trace_off, _ = await _run(timed_event_runner, False)
    on, trace_on, retr = await _run(timed_event_runner, True)
    assert _comparable(on) == _comparable(off)
    spec = [e["speculation"] for e in trace_on if e["event"] == "output_emitted"]
    assert spec and spec[0]["launched"] >= 1 and spec[0]["reused"] >= 1
    # the speculated text was searched once, during speech - not again at the end
    assert len(retr.texts) == len(set(retr.texts))
    # every real sub-query's retrieval telemetry is still present
    started = [e for e in trace_on if e["event"] == "retrieval_started"]
    completed = [e for e in trace_on if e["event"] == "retrieval_completed"]
    assert len(completed) >= len(started) - sum(e["event"] == "retrieval_cancelled" for e in trace_on)


async def test_speculation_can_be_disabled(timed_event_runner):
    _, trace, _ = await _run(timed_event_runner, False)
    spec = [e["speculation"] for e in trace if e["event"] == "output_emitted"]
    assert spec == [{"launched": 0, "reused": 0}]


async def test_speculative_prefetch_ids_never_collide_with_real_sub_queries(timed_event_runner):
    """Regression: the preview rolls back the controller's id counter, so a
    speculative sub-query reused the id a DIFFERENT real sub-query got later;
    that one's prefetch overwrote the speculative claim and the reuse was
    silently lost (claims recomputed at answer time: +36-42 ms)."""
    from streaming_rag.mocks import MockSynthesizer

    class RecordingSynth(MockSynthesizer):
        def __init__(self):
            super().__init__()
            self.prefetched, self.adopted = [], []

        def prefetch(self, q, r):
            self.prefetched.append(q.query_id)

        def adopt_prefetch(self, src_id, src_result, dst_id, dst_result):
            self.adopted.append((src_id, dst_id))

    config = load_config()
    synth = RecordingSynth()
    await timed_event_runner(events(), controller=RetrievalController(None, None, config),
                             retriever=CountingRetriever(), synthesizer=synth, config=config, time_scale=4.0)
    assert synth.adopted, "the speculative result was not adopted"
    real_ids = {dst for _, dst in synth.adopted}
    for src, dst in synth.adopted:
        assert "~spec" in src and src != dst
    assert not real_ids & {i for i in synth.prefetched if "~spec" in i}
