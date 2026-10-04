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


async def test_lexical_selector_KNOWN_LIMITATION_two_word_generic_phrase_can_fool_it():
    """Honestly documents a real, demonstrated boundary of the LEXICAL
    (`synthesis.selector="lexical"`, the dependency-free fallback path)
    relevance gate: a decoy sharing a two-word GENERIC PHRASE with the query
    ("approval process") — even from a genuinely different topic (travel
    approval, not catering approval) — clears both the overlap-fraction and
    the >=2-token absolute-count checks in _is_relevant, and IS asserted
    under the lexical selector. This is a real gap, not a hypothetical one:
    it was found by writing this exact adversarial case, not assumed away.
    Closing it needs topic/domain discrimination beyond token overlap — a
    real semantic signal. That signal is exactly what the CROSS-ENCODER
    selector (the default since final_report.docx's audit) adds: see
    test_cross_encoder_selector_closes_the_two_word_generic_phrase_gap
    directly below, which is the fix this docstring originally asked for.
    The lexical path is kept, and this limitation kept documented on it,
    because it is still the zero-model-download fallback."""
    cfg = load_config()
    cfg.synthesis.selector = "lexical"
    retriever = HybridRetriever(cfg, chunks=ADVERSARIAL_CHUNKS)
    await retriever.setup()
    store = SessionStore(); store.open("s1")
    synth = GroundedSynthesizer(retriever, store, cfg)

    q = SubQuery(query_id="q1", text="what is the catering approval process",
                 intent_label="catering approval", trigger="final", utterance_id="u1")
    result = _result("q1", ADVERSARIAL_CHUNKS[0])
    answer = await synth.synthesize("s1", "what is the catering approval process", [q], [result])

    # Documents CURRENT behaviour of the lexical fallback (a false positive)
    # rather than hiding it.
    assert answer.citations == ["Doc_02 §1"]
    assert answer.uncertainty is None


async def test_cross_encoder_selector_closes_the_two_word_generic_phrase_gap():
    """The default selector (cross-encoder claim selection + the learned
    refusal gate, final_report.docx §5.3/§5.8) closes the exact gap documented
    in test_lexical_selector_KNOWN_LIMITATION_... above: on the identical
    adversarial decoy, a real semantic/relevance signal (the cross-encoder's
    sentence score and the SQuAD2 reader's answer-vs-no-answer margin) now
    correctly recognises the decoy is off-topic and refuses it, instead of
    being fooled by the shared "approval process" phrase."""
    cfg = load_config()   # default: selector="cross-encoder", refusal_gate="learned"
    retriever = HybridRetriever(cfg, chunks=ADVERSARIAL_CHUNKS)
    await retriever.setup()
    store = SessionStore(); store.open("s1")
    synth = GroundedSynthesizer(retriever, store, cfg)

    q = SubQuery(query_id="q1", text="what is the catering approval process",
                 intent_label="catering approval", trigger="final", utterance_id="u1")
    result = _result("q1", ADVERSARIAL_CHUNKS[0])
    answer = await synth.synthesize("s1", "what is the catering approval process", [q], [result])

    assert answer.citations == []
    assert answer.uncertainty is not None


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


def test_speculative_claim_selection_is_reused_not_recomputed():
    from streaming_rag.config import load_config
    from streaming_rag.contracts import Chunk, Claim, EvidenceChunk, RetrievalResult, SubQuery
    from streaming_rag.session.store import SessionStore
    from streaming_rag.session.synthesis import GroundedSynthesizer
    cfg = load_config()
    cfg.synthesis.selector = "cross-encoder"
    cfg.synthesis.refusal_gate = "none"   # isolate prefetch/caching from the learned gate
    synth = GroundedSynthesizer(retriever=object(), session_store=SessionStore(), config=cfg)
    calls = []

    def fake_ce_claim(query, result):
        calls.append(query)
        return Claim(text="Tesla contracted cholera.", citations=["Doc_09 §8"])
    synth._ce_claim = fake_ce_claim
    q = SubQuery(query_id="u1.q1", text="What disease did Tesla catch?", intent_label="disease",
                 trigger="final", utterance_id="u1")
    chunk = Chunk(chunk_id="Doc_09 §8 #1", doc_id="Doc_09", section="8", text="Tesla contracted cholera.", metadata={})
    result = RetrievalResult(query_id="u1.q1", evidence=(EvidenceChunk(chunk=chunk, score=5.0, query_ids=("u1.q1",)),),
                             latency_ms=1.0, low_confidence=False)
    synth.prefetch(q, result)                       # during speech
    claims, _ = synth._claims_from([q], [result])   # at utterance end
    assert [c.text for c in claims] == ["Tesla contracted cholera."]
    assert len(calls) == 1


def test_learned_refusal_gate_suppresses_a_low_probability_claim():
    """_gated_claim must actually consult the fitted model and threshold, not
    just pass the cross-encoder's own claim through unconditionally."""
    cfg = load_config()
    cfg.synthesis.selector = "cross-encoder"
    cfg.synthesis.refusal_gate = "learned"
    synth = GroundedSynthesizer(retriever=object(), session_store=SessionStore(), config=cfg)
    claim = Claim(text="Some sentence.", citations=["Doc_00 §1"])
    synth._ce_claim = lambda query, result: claim
    synth._answer_probability = lambda query, result, claim, reader=None: 0.0   # force "refuse"
    # >= 2 content tokens: exercises the learned-model branch, not the
    # short-query lexical fallback (see test_short_query_bypasses_the_learned_model_gate)
    q = SubQuery(query_id="u1.q1", text="A real question here?", intent_label="q", trigger="final", utterance_id="u1")
    result = RetrievalResult(query_id="u1.q1", evidence=(), latency_ms=1.0, low_confidence=False)
    assert synth._gated_claim(q.text, result) is None

    synth._answer_probability = lambda query, result, claim, reader=None: 1.0   # force "answer"
    assert synth._gated_claim(q.text, result) is claim


def test_refusal_gate_threshold_matches_the_frozen_model_file():
    from streaming_rag.session.synthesis import _refusal_model
    model = _refusal_model()
    assert 0.0 < model["threshold"] < 1.0
    assert len(model["coef"]) == len(model["features"]) == len(model["scaler_mean"]) == len(model["scaler_scale"])


def test_short_query_bypasses_the_learned_model_gate():
    """A bare keyword sub-query (< 2 content tokens, no real question
    structure) is out-of-distribution for the reader/cross-encoder features
    the learned gate was fit on; it must fall back to the lexical relevance
    check instead of trusting a model score computed on text unlike anything
    it was calibrated on (see _gated_claim's comment)."""
    cfg = load_config()
    cfg.synthesis.selector = "cross-encoder"
    synth = GroundedSynthesizer(retriever=object(), session_store=SessionStore(), config=cfg)
    on_topic = Claim(text="Standard reimbursement covers economy airfare and lodging.", citations=["Doc_01 §2"])
    off_topic = Claim(text="International travel requires senior director approval.", citations=["Doc_01 §1"])
    synth._answer_probability = lambda *a: 0.0   # would refuse everything if consulted
    result = RetrievalResult(query_id="q1", evidence=(), latency_ms=1.0, low_confidence=False)

    synth._ce_claim = lambda query, result: on_topic
    assert synth._gated_claim("reimbursement", result) is on_topic   # 1 content token: lexical fallback, overlaps -> kept

    synth._ce_claim = lambda query, result: off_topic
    assert synth._gated_claim("reimbursement", result) is None       # 1 content token: no overlap -> refused


async def test_uncertainty_note_quotes_the_sub_query_not_a_keyword_label():
    synth, store, retriever = await _synth()
    q = SubQuery(query_id="q1", text="what about parking validation", intent_label="parking validation",
                 trigger="final", utterance_id="u1")
    result = RetrievalResult(query_id="q1", evidence=(), latency_ms=1.0, low_confidence=True)
    answer = await synth.synthesize("s1", "what about parking validation", [q], [result])
    assert '"what about parking validation"' in answer.uncertainty


async def test_two_sub_queries_selecting_the_same_sentence_yield_one_claim():
    synth, store, retriever = await _synth()
    q1 = SubQuery(query_id="q1", text="reimbursement", intent_label="reimbursement", trigger="final", utterance_id="u1")
    q2 = SubQuery(query_id="q2", text="reimbursement airfare", intent_label="reimbursement airfare",
                  trigger="final", utterance_id="u1")
    answer = await synth.synthesize("s1", "reimbursement and airfare reimbursement",
                                    [q1, q2], [_result("q1", CHUNKS[1]), _result("q2", CHUNKS[1])])
    assert len(answer.claims) == 1
    assert answer.citations == ["Doc_01 §2"]


# ---------- synthesis.anaphoric_context: dangling-reference sentences get their predecessor ----------
async def _context_setup(flag: bool):
    synth, store, retriever = await _synth()
    synth._config.synthesis.anaphoric_context = flag
    chunk = Chunk(chunk_id="Doc_09 §1 #1", doc_id="Doc_09", section="1",
                  text="The company was founded in Pune in 1802. It is now based in Mumbai. The venue holds 200 people.")
    return synth, chunk, _result("q1", chunk)


async def test_anaphoric_context_is_on_by_default_and_can_be_switched_off():
    from streaming_rag.config import SynthesisConfig
    assert SynthesisConfig().anaphoric_context is True
    synth, chunk, result = await _context_setup(False)
    claim = Claim(text="It is now based in Mumbai.", citations=[chunk.citation])
    assert synth._with_context(claim, result).text == "It is now based in Mumbai."


async def test_anaphoric_context_prepends_the_previous_sentence():
    synth, chunk, result = await _context_setup(True)
    claim = Claim(text="It is now based in Mumbai.", citations=[chunk.citation])
    out = synth._with_context(claim, result)
    assert out.text == "The company was founded in Pune in 1802. It is now based in Mumbai."
    assert out.citations == [chunk.citation]          # same source, nothing invented


async def test_anaphoric_context_leaves_standalone_and_first_sentences_alone():
    synth, chunk, result = await _context_setup(True)
    standalone = Claim(text="The venue holds 200 people.", citations=[chunk.citation])
    assert synth._with_context(standalone, result).text == "The venue holds 200 people."
    first = Claim(text="The company was founded in Pune in 1802.", citations=[chunk.citation])
    assert synth._with_context(first, result).text == first.text
    # a dangling sentence with nothing before it in the chunk cannot be repaired, and must not crash
    lone = Chunk(chunk_id="Doc_09 §2 #1", doc_id="Doc_09", section="2", text="It is now based in Mumbai.")
    lone_claim = Claim(text="It is now based in Mumbai.", citations=[lone.citation])
    assert synth._with_context(lone_claim, _result("q2", lone)).text == "It is now based in Mumbai."
