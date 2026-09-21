from __future__ import annotations

from streaming_rag.config import load_config
from streaming_rag.contracts import Chunk, EvidenceChunk, RetrievalResult, SubQuery
from streaming_rag.retrieval import HybridRetriever
from streaming_rag.session.grounding import verify, claim_support
from streaming_rag.session.store import SessionStore
from streaming_rag.session.synthesis import GroundedSynthesizer
from streaming_rag.contracts import Claim


CHUNKS = [
    Chunk(chunk_id="Doc_01 §1 #1", doc_id="Doc_01", section="1",
          text="International travel requires senior director approval before booking."),
    Chunk(chunk_id="Doc_01 §2 #1", doc_id="Doc_01", section="2",
          text="Standard reimbursement covers economy airfare and lodging."),
]


async def _synth():
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=CHUNKS)
    await retriever.setup()  # builds _by_citation, needed by the grounding verifier
    store = SessionStore(); store.open("s1")
    return GroundedSynthesizer(retriever, store, cfg), store, retriever


def _result(query_id, chunk, score=0.9):
    return RetrievalResult(
        query_id=query_id,
        evidence=(EvidenceChunk(chunk=chunk, score=score, query_ids=(query_id,)),),
        latency_ms=1.0, low_confidence=False,
    )


async def test_synthesize_produces_grounded_claim_with_real_citation():
    synth, store, retriever = await _synth()
    q = SubQuery(query_id="q1", text="reimbursement", intent_label="reimbursement", trigger="final", utterance_id="u1")
    result = _result("q1", CHUNKS[1])
    answer = await synth.synthesize("s1", "summarize reimbursement", [q], [result])
    assert answer.version == 1
    assert answer.citations == ["Doc_01 §2"]
    assert answer.claims[0].supported is True


async def test_synthesize_flags_uncertainty_on_empty_evidence():
    synth, store, retriever = await _synth()
    q = SubQuery(query_id="q1", text="parking validation", intent_label="parking", trigger="final", utterance_id="u1")
    result = RetrievalResult(query_id="q1", evidence=(), latency_ms=1.0, low_confidence=True)
    answer = await synth.synthesize("s1", "what about parking", [q], [result])
    assert answer.uncertainty is not None
    assert answer.citations == []


async def test_refine_keeps_prior_citations_and_increments_version():
    synth, store, retriever = await _synth()
    q1 = SubQuery(query_id="q1", text="reimbursement", intent_label="reimbursement", trigger="final", utterance_id="u1")
    v1 = await synth.synthesize("s1", "summarize reimbursement", [q1], [_result("q1", CHUNKS[1])])
    store.record_answer("s1", v1)

    q2 = SubQuery(query_id="q2", text="international travel approval", intent_label="international",
                   trigger="refinement", utterance_id="u2")
    v2 = await synth.refine("s1", "actually it was international", [q2], [_result("q2", CHUNKS[0])])

    assert v2.version == 2
    assert v2.parent_version == 1
    assert set(v1.citations) <= set(v2.citations)


async def test_restructure_uses_no_new_citations():
    synth, store, retriever = await _synth()
    q1 = SubQuery(query_id="q1", text="reimbursement", intent_label="reimbursement", trigger="final", utterance_id="u1")
    v1 = await synth.synthesize("s1", "summarize reimbursement", [q1], [_result("q1", CHUNKS[1])])
    store.record_answer("s1", v1)

    v2 = await synth.restructure("s1", "please repeat that in two bullets")
    assert v2.version == 2
    assert v2.change_kind == "restructure"
    assert set(v2.citations) == set(v1.citations)


async def test_grounding_verify_strips_fabricated_citation():
    retriever = HybridRetriever(load_config(), chunks=CHUNKS)
    await retriever.setup()

    claims = [Claim(text="International travel requires senior director approval",
                     citations=["Doc_999 §1"])]
    report = verify(claims, retriever.get_chunk_by_citation)
    assert report.fabricated_citations == ["Doc_999 §1"]
    assert report.claims == []  # nothing survives when every citation is fake


def test_claim_support_measures_token_overlap():
    high = claim_support("senior director approval required", "senior director approval is required for this")
    low = claim_support("catering vendor pricing", "senior director approval is required for this")
    assert high > low
