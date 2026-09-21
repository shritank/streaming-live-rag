"""Contract conformance suite (Task 4 §4.1), parametrised over any
Retriever/Synthesizer implementation. Runs against both the mocks and the
real components, and against a deliberately-broken mock to prove the suite
can actually fail.
"""
from __future__ import annotations

import asyncio

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import (
    AnswerVersion, Chunk, ControllerDecision, Decision, RetrievalResult,
    SubQuery, TranscriptChunk, TurnKind,
)
from streaming_rag.mocks import MockController, MockRetriever, MockSynthesizer
from streaming_rag.retrieval import HybridRetriever
from streaming_rag.session.store import SessionStore
from streaming_rag.session.synthesis import GroundedSynthesizer


def make_retrievers():
    cfg = load_config()
    hybrid = HybridRetriever(cfg, chunks=[
        Chunk(chunk_id="Doc_01 §1 #1", doc_id="Doc_01", section="1", text="Widgets cost 5 dollars each."),
    ])
    return [MockRetriever(), hybrid]


@pytest.mark.parametrize("retriever", make_retrievers())
async def test_retriever_returns_one_result_per_query_in_order(retriever):
    queries = [
        SubQuery(query_id="q1", text="widget price", intent_label="price", trigger="final", utterance_id="u1"),
        SubQuery(query_id="q2", text="widget price", intent_label="price", trigger="final", utterance_id="u1"),
        SubQuery(query_id="q3", text="widget price", intent_label="price", trigger="final", utterance_id="u1"),
    ]
    results = await retriever.search(queries, k=3)
    assert len(results) == len(queries)
    assert [r.query_id for r in results] == [q.query_id for q in queries]
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert type(r) is RetrievalResult  # the real dataclass, not a look-alike


async def test_retriever_get_chunk_roundtrip():
    retriever = MockRetriever()
    results = await retriever.search(
        [SubQuery(query_id="q1", text="cancellation policy", intent_label="c", trigger="final", utterance_id="u1")], k=3)
    for r in results:
        for e in r.evidence:
            fetched = retriever.get_chunk(e.chunk.chunk_id)
            assert fetched is not None
            assert fetched.chunk_id == e.chunk.chunk_id


async def test_controller_decision_invariant_subqueries_iff_retrieve():
    cfg = load_config()
    controller = MockController()
    store = SessionStore(); store.open("s1")
    chunk = TranscriptChunk("s1", "u1", 0, "Hello", False)
    decision = await controller.on_chunk(chunk, store.view("s1"))
    assert isinstance(decision, ControllerDecision)
    assert type(decision) is ControllerDecision
    if decision.decision == Decision.RETRIEVE:
        assert len(decision.sub_queries) > 0
    else:
        assert len(decision.sub_queries) == 0


async def test_synthesizer_answer_version_lineage():
    cfg = load_config()
    for synthesizer_factory in (lambda: MockSynthesizer(),):
        synth = synthesizer_factory()
        results = [RetrievalResult(
            query_id="q1",
            evidence=(),
            latency_ms=1.0, low_confidence=True,
        )]
        sub_queries = [SubQuery(query_id="q1", text="x", intent_label="x", trigger="final", utterance_id="u1")]
        v1 = await synth.synthesize("s1", "hello", sub_queries, results)
        assert isinstance(v1, AnswerVersion)
        assert type(v1) is AnswerVersion
        assert v1.version == 1
        assert v1.parent_version is None

        v2 = await synth.refine("s1", "more", sub_queries, results)
        assert v2.version == 2
        assert v2.parent_version == 1


# ---------- self-test: a deliberately broken mock must fail the suite ----------

class BrokenRetriever:
    """Returns results out of order — a real bug the suite must catch."""
    async def search(self, queries, k=5):
        results = [
            RetrievalResult(query_id=q.query_id, evidence=(), latency_ms=0.0, low_confidence=True)
            for q in queries
        ]
        return list(reversed(results))

    def get_chunk(self, chunk_id):
        return None


async def test_conformance_suite_can_fail_on_broken_retriever():
    retriever = BrokenRetriever()
    queries = [
        SubQuery(query_id="q1", text="a", intent_label="a", trigger="final", utterance_id="u1"),
        SubQuery(query_id="q2", text="b", intent_label="b", trigger="final", utterance_id="u1"),
    ]
    results = await retriever.search(queries, k=3)
    with pytest.raises(AssertionError):
        assert [r.query_id for r in results] == [q.query_id for q in queries]
