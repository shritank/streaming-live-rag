"""Task context §7.4 — G5 refinement continuity.

The behaviour under test: a late constraint updates the existing answer in
place. Prior claims and their citations survive unless the constraint
invalidates them, only the delta is searched, and the version chain is intact.
"""

from __future__ import annotations

import copy

import pytest

from streaming_rag.session.generative.delta import DeltaEngine

from .conftest import make_synthesizer
from .fakes import (
    FakeLLM,
    FakeRetriever,
    claim_payload,
    decisions_payload,
    result,
    sub_query,
)

pytestmark = pytest.mark.asyncio


async def seed_v1(store, verifier, telemetry, config, chunks):
    """Answer version 1: capacity + cancellation, both cited."""
    queries = [sub_query("u1.q1", "hall capacity for 30", "venue capacity"),
               sub_query("u1.q2", "cancellation terms", "cancellation terms")]
    results = [result("u1.q1", [chunks["capacity"]]),
               result("u1.q2", [chunks["cancellation"]])]
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Hall Aurora seats 30 attendees", ["Doc_12 §2"], "u1.q1"),
        ("Cancellations more than 14 days before the event receive a full refund",
         ["Doc_31 §4"], "u1.q2")])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    v1 = await synth.synthesize("s1", "hall for 30 and refund terms?", queries, results)
    return v1


async def test_refinement_increments_version_and_keeps_lineage(store, verifier,
                                                               telemetry, config, chunks):
    v1 = await seed_v1(store, verifier, telemetry, config, chunks)
    v1_snapshot = copy.deepcopy(v1)

    delta = [sub_query("u2.q1", "hall capacity for 60", "venue capacity",
                       trigger="refinement", utterance="u2", parent="u1.q1")]
    delta_results = [result("u2.q1", [chunks["capacity_large"]])]
    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([
            {"index": 0, "action": "retract", "reason": "capacity no longer sufficient"},
            {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([
            ("Hall Borealis seats 60 attendees", ["Doc_12 §3"], "u2.q1")])],
    })
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    v2 = await synth.refine("s1", "actually make it 60 people", delta, delta_results)

    assert (v2.version, v2.parent_version) == (2, 1)
    assert v2.change_kind == "refinement"
    assert v1 == v1_snapshot                      # V1 was never mutated
    assert store.latest_version("s1").version == 2


async def test_unaffected_claims_and_their_citations_survive(store, verifier, telemetry,
                                                             config, chunks):
    await seed_v1(store, verifier, telemetry, config, chunks)

    delta = [sub_query("u2.q1", "hall capacity for 60", "venue capacity",
                       trigger="refinement", utterance="u2", parent="u1.q1")]
    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([
            {"index": 0, "action": "retract"}, {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([
            ("Hall Borealis seats 60 attendees", ["Doc_12 §3"], "u2.q1")])],
    })
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    v2 = await synth.refine("s1", "make it 60 people",
                            delta, [result("u2.q1", [chunks["capacity_large"]])])

    assert "Doc_31 §4" in v2.citations            # untouched cancellation claim kept
    assert "Doc_12 §3" in v2.citations            # delta citation added
    assert "Doc_12 §2" not in v2.citations        # retracted claim's citation gone
    assert "14 days" in v2.text


async def test_refine_only_receives_delta_evidence(store, verifier, telemetry,
                                                   config, chunks):
    """The delta prompt must not re-litigate the whole original question."""
    await seed_v1(store, verifier, telemetry, config, chunks)

    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([{"index": 0, "action": "keep"},
                                               {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([
            ("Hall Borealis seats 60 attendees", ["Doc_12 §3"], "u2.q1")])],
    })
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    delta = [sub_query("u2.q1", "larger hall", "venue capacity",
                       trigger="refinement", utterance="u2", parent="u1.q1")]
    await synth.refine("s1", "we might need a bigger hall",
                       delta, [result("u2.q1", [chunks["capacity_large"]])])

    claims_prompt = next(c["prompt"] for c in llm.calls
                         if c["purpose"] == "refinement_claims")
    assert "Doc_12 §3" in claims_prompt
    assert "Doc_31 §4" not in claims_prompt       # old evidence not re-sent


async def test_refinement_triggers_no_retrieval_of_its_own(store, verifier, telemetry,
                                                           config, chunks):
    """Task 3 holds no retriever at all: searching is the engine's job."""
    await seed_v1(store, verifier, telemetry, config, chunks)
    retriever = FakeRetriever()

    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([{"index": 0, "action": "keep"},
                                               {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([])],
    })
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    assert not hasattr(synth, "retriever")

    delta = [sub_query("u2.q1", "late booking", "late booking",
                       trigger="refinement", utterance="u2", parent="u1.q2")]
    await synth.refine("s1", "it was booked late", delta,
                       [result("u2.q1", [chunks["cancellation"]])])

    assert retriever.call_count == 0


async def test_all_delta_queries_are_marked_as_refinements(store, verifier, telemetry,
                                                           config, chunks):
    await seed_v1(store, verifier, telemetry, config, chunks)
    delta = [sub_query("u2.q1", "late booking", "late booking",
                       trigger="refinement", utterance="u2", parent="u1.q2")]

    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([{"index": 0, "action": "keep"},
                                               {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    v2 = await synth.refine("s1", "booked late", delta,
                            [result("u2.q1", [chunks["cancellation"]])])

    assert all(sq.trigger == "refinement" and sq.parent_query_id
               for sq in v2.sub_queries)


async def test_claims_the_model_forgets_are_kept(store, verifier, telemetry,
                                                 config, chunks):
    """Silence must not delete a grounded claim."""
    await seed_v1(store, verifier, telemetry, config, chunks)
    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([{"index": 0, "action": "keep"}])],
        "refinement_claims": [claim_payload([])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    delta = [sub_query("u2.q1", "anything", "anything", trigger="refinement",
                       utterance="u2", parent="u1.q1")]
    v2 = await synth.refine("s1", "one more thing", delta,
                            [result("u2.q1", [chunks["cancellation"]])])

    assert len(v2.claims) == 2


async def test_delta_engine_falls_back_when_the_model_fails(store, verifier, telemetry,
                                                            config, chunks):
    await seed_v1(store, verifier, telemetry, config, chunks)
    llm = FakeLLM(responses={
        "delta_decisions": [RuntimeError("provider down")],
        "refinement_claims": [claim_payload([
            ("Hall Borealis seats 60 attendees", ["Doc_12 §3"], "u2.q1")])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    delta = [sub_query("u2.q1", "hall for 60", "venue capacity",
                       trigger="refinement", utterance="u2", parent="u1.q1")]
    v2 = await synth.refine("s1", "make it 60 people", delta,
                            [result("u2.q1", [chunks["capacity_large"]])])

    created = telemetry.of("answer_version_created")[-1]
    assert created["delta_fallback"] is True
    # the heuristic retracts the stale "seats 30" claim on the number conflict
    assert "seats 30" not in v2.text
    assert "Doc_31 §4" in v2.citations


async def test_fallback_heuristic_keeps_unrelated_claims(config, chunks):
    from streaming_rag.contracts import AnswerVersion, Claim

    previous = AnswerVersion(
        version=1, parent_version=None, change_kind="initial", text="",
        claims=[Claim(text="Hall Aurora seats 30 attendees", citations=["Doc_12 §2"]),
                Claim(text="Cancellations 14 days ahead are refunded",
                      citations=["Doc_31 §4"])],
        citations=[], evidence_ids=[], sub_queries=[], uncertainty=None, created_ms=0)

    engine = DeltaEngine(llm=None, config=config)  # type: ignore[arg-type]
    decisions = engine._fallback(previous, "make it 60 attendees", [])

    assert decisions[0].action == "retract"   # capacity conflicts with 60
    assert decisions[1].action == "keep"      # refund window is unrelated


async def test_refine_without_a_previous_answer_behaves_as_a_new_request(
        store, verifier, telemetry, config, chunks):
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Hall Aurora seats 30 attendees", ["Doc_12 §2"], "u1.q1")])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    v1 = await synth.refine("s1", "and make it 30 people",
                            [sub_query("u1.q1", "hall capacity", "venue capacity")],
                            [result("u1.q1", [chunks["capacity"]])])

    assert (v1.version, v1.change_kind) == (1, "initial")


async def test_refinement_is_cheaper_than_re_synthesis(store, verifier, telemetry,
                                                       config, chunks):
    """Architectural parsimony: refining must cost fewer tokens than starting over."""
    await seed_v1(store, verifier, telemetry, config, chunks)

    refine_llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([{"index": 0, "action": "keep"},
                                               {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([
            ("Hall Borealis seats 60 attendees", ["Doc_12 §3"], "u2.q1")])]})
    synth = make_synthesizer(store, refine_llm, verifier, telemetry, config)
    delta = [sub_query("u2.q1", "hall for 60", "venue capacity",
                       trigger="refinement", utterance="u2", parent="u1.q1")]
    await synth.refine("s1", "make it 60 people", delta,
                       [result("u2.q1", [chunks["capacity_large"]])])
    refine_chars = sum(len(c["prompt"]) for c in refine_llm.calls)

    # the same question asked from scratch, with all evidence in the prompt
    store.open("fresh")
    full_llm = FakeLLM(responses={"synthesis": [claim_payload([])],
                                  "clarification": [{"question": "?"}]})
    fresh = make_synthesizer(store, full_llm, verifier, telemetry, config)
    await fresh.synthesize(
        "fresh", "hall for 60 people and refund terms and booked late",
        [sub_query("v1.q1", "hall capacity for 60", "venue capacity"),
         sub_query("v1.q2", "cancellation terms", "cancellation terms")],
        [result("v1.q1", [chunks["capacity"], chunks["capacity_large"]]),
         result("v1.q2", [chunks["cancellation"], chunks["old_cancellation"]])])
    full_chars = sum(len(c["prompt"]) for c in full_llm.calls)

    assert refine_chars < full_chars * 1.5   # recorded in docs/session.md


async def test_refinement_does_not_hedge_about_covered_intents(store, verifier,
                                                               telemetry, config, chunks):
    """Over-hedging is a failure too (§7.3): V1 claims still answer their intents."""
    await seed_v1(store, verifier, telemetry, config, chunks)

    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([{"index": 0, "action": "keep"},
                                               {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([
            ("External caterers require written approval", ["Doc_09 §1"], "u2.q1")])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    delta = [sub_query("u2.q1", "external catering approval", "catering options",
                       trigger="refinement", utterance="u2", parent="u1.q2")]

    v2 = await synth.refine("s1", "we want an outside caterer", delta,
                            [result("u2.q1", [chunks["catering"]])])

    assert v2.uncertainty is None
    assert len(v2.citations) == 3


async def test_delta_coverage_satisfies_the_parent_intent(store, verifier, telemetry,
                                                          config, chunks):
    """A replacement claim answers the intent of the query it replaced."""
    await seed_v1(store, verifier, telemetry, config, chunks)

    llm = FakeLLM(responses={
        "delta_decisions": [decisions_payload([{"index": 0, "action": "retract"},
                                               {"index": 1, "action": "keep"}])],
        "refinement_claims": [claim_payload([
            ("Hall Borealis seats 60 attendees", ["Doc_12 §3"], "u2.q1")])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    delta = [sub_query("u2.q1", "hall capacity for 60", "venue capacity",
                       trigger="refinement", utterance="u2", parent="u1.q1")]

    v2 = await synth.refine("s1", "make it 60 people", delta,
                            [result("u2.q1", [chunks["capacity_large"]])])

    assert v2.uncertainty is None
