"""RetrievalController.preview_final: the speculative final pass must leave the
controller exactly as it was and predict the real final pass's sub-queries."""
from __future__ import annotations

import copy

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import Decision, TranscriptChunk
from streaming_rag.controller import RetrievalController
from streaming_rag.session.store import SessionStore

UTTERANCES = [
    ["What came into force after the", " new constitution was herald?, What attracts", " the tourists to Kenya?"],
    ["I need to plan a customer workshop in", " Pune for 30 people, and I need",
     " the cancellation policy and the catering options."],
    ["how many tons of saharan dust", " falls on the amazon basin each year"],   # unpunctuated, ASR-like
    ["Before which event did Chagatai", " publicly dispute Jochi's paternity"],   # clause still open
]


def _state(c: RetrievalController, uid: str):
    return copy.deepcopy([d.get(("s1", uid)) for d in (c._accumulated, c._previous, c._covered, c._issued,
                                               c._live, c._clauses)])


@pytest.mark.parametrize("chunks", UTTERANCES)
async def test_preview_is_side_effect_free_and_matches_the_real_final_pass(chunks):
    c = RetrievalController(None, None, load_config())
    store = SessionStore()
    store.open("s1")
    for i, text in enumerate(chunks):
        await c.on_chunk(TranscriptChunk("s1", "u1", i * 800, text, False), store.view("s1"))
        before = _state(c, "u1")
        preview = await c.preview_final("u1", "s1", store.view("s1"))
        assert _state(c, "u1") == before, "preview_final must not change controller state"
    real = await c.on_chunk(TranscriptChunk("s1", "u1", 9999, "", True), store.view("s1"))
    assert preview.decision == real.decision
    assert [q.text for q in preview.sub_queries] == [q.text for q in real.sub_queries]


async def test_preview_on_an_unknown_utterance_leaves_no_trace():
    c = RetrievalController(None, None, load_config())
    store = SessionStore()
    store.open("s1")
    decision = await c.preview_final("never-seen", "s1", store.view("s1"))
    assert decision.decision in (Decision.WAIT, Decision.RETRIEVE, Decision.SUPPRESS)
    assert all(("s1", "never-seen") not in d for d in (c._accumulated, c._previous, c._covered, c._issued,
                                                       c._live, c._clauses))


@pytest.mark.parametrize("chunks", UTTERANCES)
async def test_preview_with_pending_text_matches_that_chunk_then_the_final_pass(chunks):
    """ASR-hypothesis speculation: preview(committed, pending=last chunk) must
    predict exactly what the real last chunk + final pass issue."""
    c = RetrievalController(None, None, load_config())
    store = SessionStore()
    store.open("s1")
    for i, text in enumerate(chunks[:-1]):
        await c.on_chunk(TranscriptChunk("s1", "u1", i * 800, text, False), store.view("s1"))
    before = _state(c, "u1")
    preview = await c.preview_final("u1", "s1", store.view("s1"), pending_text=chunks[-1])
    assert _state(c, "u1") == before
    last = await c.on_chunk(TranscriptChunk("s1", "u1", 5000, chunks[-1], False), store.view("s1"))
    final = await c.on_chunk(TranscriptChunk("s1", "u1", 9999, "", True), store.view("s1"))
    real = [q.text for d in (last, final) if d.decision == Decision.RETRIEVE for q in d.sub_queries]
    assert [q.text for q in preview.sub_queries] == real
