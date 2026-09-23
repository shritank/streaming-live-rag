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


# ---------------------------------------------------------------------------
# Adversarial / hard grounding tests.
#
# grounding.verify()'s post-hoc claim_support check is near-tautological for
# this system BY DESIGN: because synthesis is extractive (a claim's text is
# always a verbatim sentence taken from the chunk it cites), that claim will
# always self-support against its own citation — it proves a citation is not
# invented, not that the retrieved chunk actually answers the question. The
# real defence against asserting a wrong-topic-but-superficially-matching
# passage is the query-relevance gate in _claims_from/_is_relevant, exercised
# through the REAL GroundedSynthesizer (not a mock), with deliberately
# adversarial decoy evidence. These tests prove that gate holds, rather than
# assuming it does.
# ---------------------------------------------------------------------------

ADVERSARIAL_CHUNKS = [
    # Shares "approval"/"process" with a catering query, but is genuinely
    # about travel — a distractor a naive keyword system would accept.
    Chunk(chunk_id="Doc_02 §1 #1", doc_id="Doc_02", section="1",
          text="The travel approval process requires the traveller's manager to sign off before any booking is made."),
    # Mentions "Tesla" in nearly every sentence (the exact ubiquitous-entity
    # trap _is_relevant is designed to catch) without ever discussing
    # nationality, which is what the adversarial query below asks about.
    Chunk(chunk_id="Doc_03 §1 #1", doc_id="Doc_03", section="1",
          text="Tesla filed the patent in 1891. Tesla later demonstrated the device at a public exhibition in New York."),
]


async def test_query_relevance_gate_rejects_single_shared_generic_word_decoy():
    """A single sub-query whose ONLY available evidence shares just one
    generic word with the query ("approval") but is about a different topic
    (travel, not catering vendors) must be flagged uncertain, never
    asserted — even though the decoy would trivially self-support in a
    post-hoc check, since its text is extracted verbatim from itself."""
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=ADVERSARIAL_CHUNKS)
    await retriever.setup()
    store = SessionStore(); store.open("s1")
    synth = GroundedSynthesizer(retriever, store, cfg)

    q = SubQuery(query_id="q1", text="which vendor needs catering approval",
                 intent_label="catering vendor approval", trigger="final", utterance_id="u1")
    result = _result("q1", ADVERSARIAL_CHUNKS[0])  # the only evidence IS the wrong-topic decoy
    answer = await synth.synthesize("s1", "which vendor needs catering approval", [q], [result])

    assert answer.citations == []
    assert answer.uncertainty is not None


async def test_KNOWN_LIMITATION_two_word_generic_phrase_overlap_can_fool_the_gate():
    """Honestly documents a real, demonstrated boundary of the lexical
    relevance gate: a decoy sharing a two-word GENERIC PHRASE with the query
    ("approval process") — even from a genuinely different topic (travel
    approval, not catering approval) — clears both the overlap-fraction and
    the >=2-token absolute-count checks in _is_relevant, and IS currently
    asserted. This is a real gap, not a hypothetical one: it was found by
    writing this exact adversarial case, not assumed away. Closing it would
    need topic/domain discrimination beyond token overlap (e.g. a real
    semantic or entity-aware signal) — attempts at that during development
    traded this class of miss for a *worse* one (a completely unrelated
    document winning a similarity tie-break) and were reverted; see
    docs/benchmark_report.md §0e for the full account. This test exists so
    a future fix is verified against a real failing case, and so this
    boundary is never silently forgotten."""
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=ADVERSARIAL_CHUNKS)
    await retriever.setup()
    store = SessionStore(); store.open("s1")
    synth = GroundedSynthesizer(retriever, store, cfg)

    q = SubQuery(query_id="q1", text="what is the catering approval process",
                 intent_label="catering approval", trigger="final", utterance_id="u1")
    result = _result("q1", ADVERSARIAL_CHUNKS[0])
    answer = await synth.synthesize("s1", "what is the catering approval process", [q], [result])

    # Documents CURRENT behaviour (a false positive) rather than hiding it.
    # If this assertion ever flips to citations == [], the gate has
    # genuinely improved — update this test to match, don't just delete it.
    assert answer.citations == ["Doc_02 §1"]
    assert answer.uncertainty is None


async def test_query_relevance_gate_rejects_ubiquitous_entity_without_topic_match():
    """A chunk that repeats the query's named entity constantly, but never
    touches the actual attribute being asked about, must not be asserted."""
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=ADVERSARIAL_CHUNKS)
    await retriever.setup()
    store = SessionStore(); store.open("s1")
    synth = GroundedSynthesizer(retriever, store, cfg)

    q = SubQuery(query_id="q1", text="what nationality was Tesla",
                 intent_label="Tesla nationality", trigger="final", utterance_id="u1")
    result = _result("q1", ADVERSARIAL_CHUNKS[1])
    answer = await synth.synthesize("s1", "what nationality was Tesla", [q], [result])

    assert answer.citations == []
    assert answer.uncertainty is not None


async def test_extractive_self_support_is_a_known_property_not_a_safety_net():
    """Documents the exact trade-off: an extractive claim ALWAYS self-supports
    against the chunk it was extracted from (grounding.verify alone would
    pass a wrong-topic decoy). This is expected — it is why _is_relevant,
    not claim_support, is the gate that must reject bad evidence (see the
    two tests above). This test pins that expectation down so a future
    change to claim_support's scoring can't silently start relying on it as
    a topic-relevance check."""
    decoy = ADVERSARIAL_CHUNKS[0]
    claim = Claim(text=decoy.text, citations=[decoy.citation])
    # Self-support against an unrelated query's evidence is high, by
    # construction, even though the chunk never mentions catering at all.
    self_support = claim_support(claim.text, decoy.text)
    assert self_support > 0.9


async def test_low_confidence_evidence_never_becomes_a_claim():
    """A retriever that flags its own best result as low_confidence must be
    trusted immediately — no relevance scan, no fallback, no assertion."""
    cfg = load_config()
    retriever = HybridRetriever(cfg, chunks=ADVERSARIAL_CHUNKS)
    await retriever.setup()
    store = SessionStore(); store.open("s1")
    synth = GroundedSynthesizer(retriever, store, cfg)

    q = SubQuery(query_id="q1", text="anything", intent_label="anything",
                 trigger="final", utterance_id="u1")
    result = RetrievalResult(
        query_id="q1",
        evidence=(EvidenceChunk(chunk=ADVERSARIAL_CHUNKS[0], score=0.99, query_ids=("q1",)),),
        latency_ms=1.0, low_confidence=True,  # the retriever itself doesn't trust this
    )
    answer = await synth.synthesize("s1", "anything", [q], [result])
    assert answer.citations == []
    assert answer.uncertainty is not None
