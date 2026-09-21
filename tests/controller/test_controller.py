from __future__ import annotations

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import Decision, TranscriptChunk, TurnKind
from streaming_rag.controller import RetrievalController
from streaming_rag.session.store import SessionStore


@pytest.fixture
def controller():
    return RetrievalController(None, None, load_config())


@pytest.fixture
def session():
    store = SessionStore()
    store.open("s1")
    return store


async def test_incomplete_fragment_waits(controller, session):
    chunk = TranscriptChunk("s1", "u1", 0, "I need to", False)
    decision = await controller.on_chunk(chunk, session.view("s1"))
    assert decision.decision == Decision.WAIT


async def test_entity_bearing_fragment_retrieves(controller, session):
    chunk = TranscriptChunk("s1", "u1", 0, "I need to plan a workshop in Pune for 30 people", False)
    decision = await controller.on_chunk(chunk, session.view("s1"))
    assert decision.decision == Decision.RETRIEVE
    assert len(decision.sub_queries) >= 1


async def test_multi_intent_utterance_decomposes(controller, session):
    chunk = TranscriptChunk(
        "s1", "u1", 0,
        "I need a workshop in Pune, the cancellation policy, and the catering options.",
        True)
    decision = await controller.on_chunk(chunk, session.view("s1"))
    assert decision.decision == Decision.RETRIEVE
    assert len(decision.sub_queries) >= 2
    labels = {q.intent_label for q in decision.sub_queries}
    assert len(labels) == len(decision.sub_queries)  # no duplicate intents


async def test_never_reemits_a_semantically_equivalent_subquery(controller, session):
    text = "the cancellation policy for Pune"
    chunk1 = TranscriptChunk("s1", "u1", 0, text, False)
    d1 = await controller.on_chunk(chunk1, session.view("s1"))
    seen_ids = {q.query_id for q in d1.sub_queries}

    chunk2 = TranscriptChunk("s1", "u1", 500, "", True)  # nothing new
    d2 = await controller.on_chunk(chunk2, session.view("s1"))
    assert d2.decision != Decision.RETRIEVE or not (
        {q.query_id for q in d2.sub_queries} & seen_ids
    )


async def test_presentation_only_suppresses_after_prior_answer(controller):
    store = SessionStore()
    store.open("s1")

    class FakeView:
        session_id = "s1"
        def has_answer(self): return True
        def latest_answer(self): return None
        def topic_summary(self): return "cancellation policy"
        def prior_sub_queries(self): return []

    chunk = TranscriptChunk("s1", "u2", 0, "Please repeat your last answer in two bullets.", False)
    decision = await controller.on_chunk(chunk, FakeView())
    assert decision.decision == Decision.SUPPRESS
    assert decision.turn_kind == TurnKind.PRESENTATION_ONLY


async def test_chit_chat_suppresses():
    controller = RetrievalController(None, None, load_config())
    store = SessionStore(); store.open("s1")
    chunk = TranscriptChunk("s1", "u1", 0, "Hi", False)
    decision = await controller.on_chunk(chunk, store.view("s1"))
    assert decision.decision == Decision.SUPPRESS
    assert decision.turn_kind == TurnKind.CHIT_CHAT
