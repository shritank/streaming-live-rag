"""turn_result.sub_queries is part of the external output contract: it must be
the utterance's FINAL decomposition, not every provisional fragment the
controller issued on the way (regression: a superseded fragment such as
"universal band that digital receivers will" was reported next to the real
question, i.e. the guide's over-fragmenting pitfall, while synthesis - and every
trace-based scorer - correctly ignored it)."""
from __future__ import annotations

from streaming_rag.config import load_config
from streaming_rag.controller import RetrievalController
from streaming_rag.mocks import MockRetriever


def guide_format_events():
    # literal guide §6.6 shape: chunk payloads carry no session_id
    return [
        {"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}},
        {"timestamp_ms": 0, "event_type": "transcript_chunk",
         "payload": {"utterance_id": "u1", "text": "What is the universal band that digital receivers will"}},
        {"timestamp_ms": 800, "event_type": "transcript_chunk",
         "payload": {"utterance_id": "u1",
                     "text": " receive free-to-air channels on, and what does the Wibjorn Karlen graph show?"}},
        {"timestamp_ms": 2100, "event_type": "utterance_end", "payload": {"utterance_id": "u1"}},
        {"timestamp_ms": 9000, "event_type": "session_end", "payload": {}},
    ]


async def test_turn_result_lists_only_live_sub_queries(timed_event_runner):
    config = load_config()
    results, trace = await timed_event_runner(guide_format_events(),
                                              controller=RetrievalController(None, None, config),
                                              retriever=MockRetriever(latency_ms=30.0), config=config,
                                              time_scale=4.0)
    emitted = {qid: text for e in trace if e["event"] == "sub_queries_emitted"
               for qid, text in zip(e["query_ids"], e["queries"])}
    cancelled = {qid for e in trace if e["event"] == "retrieval_cancelled" for qid in e["query_ids"]}
    assert cancelled, "the scenario must actually supersede a provisional fragment"
    live = [t for qid, t in emitted.items() if qid not in cancelled]
    assert results[0]["sub_queries"] == live
    assert len(live) == 2 and not any(t in results[0]["sub_queries"] for t in
                                      (emitted[q] for q in cancelled))
    # the trigger log still records the provisional retrieval (it happened)
    assert [ev["trigger"] for ev in results[0]["retrieval_events"]][0] == "provisional"
