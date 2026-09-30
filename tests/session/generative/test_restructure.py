"""Task context §7.5 — presentation-only turns.

The contract: reformatting touches presentation and nothing else. No retrieval,
no new facts, no new citations. This is the answer side of the controller's
SUPPRESS decision (guide Example 3).
"""

from __future__ import annotations

import pytest

from .conftest import make_synthesizer
from .fakes import FakeLLM, FakeRetriever, claim_payload, result, sub_query

pytestmark = pytest.mark.asyncio


async def seed(store, verifier, telemetry, config, chunks):
    queries = [sub_query("u1.q1", "hall capacity", "venue capacity"),
               sub_query("u1.q2", "cancellation terms", "cancellation terms")]
    results = [result("u1.q1", [chunks["capacity"]]),
               result("u1.q2", [chunks["cancellation"]])]
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Hall Aurora seats 30 attendees", ["Doc_12 §2"], "u1.q1"),
        ("Cancellations more than 14 days before the event receive a full refund",
         ["Doc_31 §4"], "u1.q2")])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    return await synth.synthesize("s1", "hall and refunds?", queries, results)


async def test_two_bullets_without_any_retrieval(store, verifier, telemetry,
                                                 config, chunks):
    v1 = await seed(store, verifier, telemetry, config, chunks)
    retriever = FakeRetriever()

    bullets = ("- Hall Aurora seats 30 attendees. [Doc_12 §2]\n"
               "- Cancellations more than 14 days before the event receive a full "
               "refund. [Doc_31 §4]")
    llm = FakeLLM(responses={"restructure": [{"text": bullets}]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    v2 = await synth.restructure("s1", "repeat that in two bullets")

    assert retriever.call_count == 0
    assert len([line for line in v2.text.splitlines() if line.startswith("- ")]) == 2
    assert (v2.version, v2.parent_version, v2.change_kind) == (2, 1, "restructure")
    assert set(v2.citations) <= set(v1.citations)
    assert llm.purposes() == ["restructure"]          # no synthesis, no delta calls


async def test_citations_and_claims_are_carried_over_unchanged(store, verifier,
                                                               telemetry, config, chunks):
    v1 = await seed(store, verifier, telemetry, config, chunks)
    llm = FakeLLM(responses={"restructure": [{"text": "Shorter version. [Doc_12 §2]"}]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    v2 = await synth.restructure("s1", "shorter please")

    assert [c.text for c in v2.claims] == [c.text for c in v1.claims]
    assert v2.citations == v1.citations
    assert v2.evidence_ids == v1.evidence_ids


async def test_model_inventing_a_citation_cannot_add_it_to_the_record(store, verifier,
                                                                     telemetry, config,
                                                                     chunks):
    """Even if the reformat text mentions a new id, the record keeps the old set."""
    v1 = await seed(store, verifier, telemetry, config, chunks)
    llm = FakeLLM(responses={"restructure": [
        {"text": "Everything is approved. [Doc_999 §9]"}]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    v2 = await synth.restructure("s1", "summarise")

    assert "Doc_999 §9" not in v2.citations
    assert v2.citations == v1.citations


async def test_translation_keeps_the_same_invariants(store, verifier, telemetry,
                                                     config, chunks):
    v1 = await seed(store, verifier, telemetry, config, chunks)
    llm = FakeLLM(responses={"restructure": [
        {"text": "हॉल ऑरोरा में 30 लोग बैठ सकते हैं। [Doc_12 §2]"}]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    v2 = await synth.restructure("s1", "say it in Hindi")

    assert set(v2.citations) <= set(v1.citations)
    assert v2.change_kind == "restructure"


async def test_restructure_without_a_previous_answer_asks_instead_of_crashing(
        store, verifier, telemetry, config):
    llm = FakeLLM(responses={})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    version = await synth.restructure("s1", "repeat that")

    assert version.change_kind == "clarification"
    assert version.citations == []
    assert llm.calls == []


async def test_restructure_keeps_previous_text_when_the_model_fails(store, verifier,
                                                                    telemetry, config,
                                                                    chunks):
    v1 = await seed(store, verifier, telemetry, config, chunks)
    llm = FakeLLM(responses={"restructure": [RuntimeError("provider down")]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    v2 = await synth.restructure("s1", "in bullets")

    assert v2.text == v1.text
    assert telemetry.of("error")[-1]["where"] == "synthesizer.restructure"


async def test_uncertainty_note_survives_a_reformat(store, verifier, telemetry,
                                                    config, chunks):
    queries = [sub_query("u1.q1", "hall capacity", "venue capacity"),
               sub_query("u1.q2", "parking", "parking rules")]
    results = [result("u1.q1", [chunks["capacity"]]),
               result("u1.q2", [], low_confidence=True)]
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Hall Aurora seats 30 attendees", ["Doc_12 §2"], "u1.q1")])]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    v1 = await synth.synthesize("s1", "capacity and parking?", queries, results)

    llm2 = FakeLLM(responses={"restructure": [{"text": "Short form. [Doc_12 §2]"}]})
    synth2 = make_synthesizer(store, llm2, verifier, telemetry, config)
    v2 = await synth2.restructure("s1", "shorter")

    assert v2.uncertainty == v1.uncertainty
