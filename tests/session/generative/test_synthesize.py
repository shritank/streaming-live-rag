"""Task context §7.2, §7.3, §7.7 — initial synthesis, uncertainty, streaming."""

from __future__ import annotations

import asyncio

import pytest

from .conftest import make_synthesizer
from .fakes import FakeLLM, claim_payload, result, sub_query

pytestmark = pytest.mark.asyncio


async def test_answer_is_cited_and_grounded(store, llm, verifier, telemetry, config,
                                            sub_queries, results):
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    version = await synth.synthesize("s1", "hall, refunds, catering?", sub_queries, results)

    assert version.version == 1 and version.parent_version is None
    assert version.change_kind == "initial"
    assert set(version.citations) == {"Doc_12 §2", "Doc_31 §4", "Doc_09 §1"}
    for citation in version.citations:
        assert f"[{citation}]" in version.text
    assert version.uncertainty is None


async def test_unsupported_sub_intent_produces_uncertainty(store, verifier, telemetry,
                                                           config, chunks):
    """Two questions asked, only one answerable: say so instead of guessing."""
    queries = [sub_query("u1.q1", "hall capacity", "venue capacity"),
               sub_query("u1.q2", "parking rules", "parking rules")]
    results = [result("u1.q1", [chunks["capacity"]]),
               result("u1.q2", [], low_confidence=True)]
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Hall Aurora seats 30 attendees", ["Doc_12 §2"], "u1.q1")])]})

    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    version = await synth.synthesize("s1", "capacity and parking?", queries, results)

    assert version.uncertainty is not None
    assert "parking" in version.uncertainty.lower()
    assert "Not verified:" in version.text
    assert telemetry.of("uncertainty_flagged")[0]["unsupported_intents"] == ["u1.q2"]


async def test_hallucinated_claim_never_reaches_the_answer(store, verifier, telemetry,
                                                           config, chunks):
    queries = [sub_query("u1.q1", "hall capacity", "venue capacity")]
    results = [result("u1.q1", [chunks["capacity"]])]
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Hall Aurora seats 30 attendees", ["Doc_12 §2"], "u1.q1"),
        ("A free shuttle runs every 15 minutes", ["Doc_999 §9"], "u1.q1")])]})

    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    version = await synth.synthesize("s1", "capacity?", queries, results)

    assert "shuttle" not in version.text
    assert "Doc_999" not in version.text
    assert telemetry.of("grounding_checked")[0]["fabricated_citations"] == ["Doc_999 §9"]


async def test_everything_unanswerable_asks_for_clarification(store, verifier, telemetry,
                                                              config):
    queries = [sub_query("u1.q1", "gym opening hours", "gym hours")]
    results = [result("u1.q1", [], low_confidence=True)]
    llm = FakeLLM(responses={"synthesis": [claim_payload([])],
                             "clarification": [{"question": "Which site do you mean?"}]})

    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    version = await synth.synthesize("s1", "gym hours?", queries, results)

    assert version.change_kind == "clarification"
    assert version.text == "Which site do you mean?"
    assert version.citations == []


async def test_distractor_evidence_does_not_produce_claims(store, verifier, telemetry,
                                                           config, chunks):
    queries = [sub_query("u1.q1", "parking rules", "parking rules")]
    results = [result("u1.q1", [chunks["distractor"]])]
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Parking is free for all attendees", ["Doc_44 §7"], "u1.q1")])],
        "clarification": [{"question": "Which building?"}]})

    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    version = await synth.synthesize("s1", "parking?", queries, results)

    assert "free" not in version.text.lower() or version.change_kind == "clarification"


async def test_streamed_partials_reassemble_into_the_final_text(store, llm, verifier,
                                                                telemetry, config,
                                                                sub_queries, results):
    partials: list[str] = []

    async def on_partial(piece: str) -> None:
        partials.append(piece)

    synth = make_synthesizer(store, llm, verifier, telemetry, config, on_partial=on_partial)
    version = await synth.synthesize("s1", "hall?", sub_queries, results)

    assert "".join(partials) == version.text
    assert "Doc_999" not in "".join(partials)


async def test_llm_failure_degrades_instead_of_raising(store, verifier, telemetry,
                                                       config, sub_queries, results):
    llm = FakeLLM(responses={"synthesis": [RuntimeError("provider down")],
                             "clarification": [RuntimeError("also down")]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    version = await synth.synthesize("s1", "hall?", sub_queries, results)

    assert version.change_kind == "clarification"
    assert any(e["where"].startswith("synthesizer") for e in telemetry.of("error"))


async def test_malformed_model_output_is_survivable(store, verifier, telemetry, config,
                                                    sub_queries, results):
    llm = FakeLLM(responses={"synthesis": ["not json at all"],
                             "clarification": [{"question": "Could you rephrase?"}]})
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    version = await synth.synthesize("s1", "hall?", sub_queries, results)
    assert version.change_kind == "clarification"


async def test_fenced_json_is_parsed(store, verifier, telemetry, config, chunks):
    """Models often wrap JSON in a code fence; that must not lose the answer."""
    queries = [sub_query("u1.q1", "hall capacity", "venue capacity")]
    results = [result("u1.q1", [chunks["capacity"]])]
    fenced = ('```json\n{"claims": [{"text": "Hall Aurora seats 30 attendees", '
              '"citations": ["Doc_12 §2"], "sub_query_id": "u1.q1"}]}\n```')
    llm = FakeLLM(responses={"synthesis": [fenced]})

    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    version = await synth.synthesize("s1", "capacity?", queries, results)

    assert version.citations == ["Doc_12 §2"]


async def test_determinism_across_runs(store, verifier, telemetry, config,
                                       sub_queries, results, good_claims_payload):
    citations = []
    for session in ("a", "b", "c"):
        store.open(session)
        llm = FakeLLM(responses={"synthesis": [good_claims_payload]})
        synth = make_synthesizer(store, llm, verifier, telemetry, config)
        version = await synth.synthesize(session, "hall?", sub_queries, results)
        citations.append(tuple(version.citations))

    assert len(set(citations)) == 1


async def test_synthesis_does_not_block_the_event_loop(store, verifier, telemetry,
                                                       config, sub_queries, results,
                                                       good_claims_payload):
    """A slow model must not stall the loop that delivers events."""
    gaps: list[float] = []

    async def ticker() -> None:
        last = asyncio.get_running_loop().time()
        for _ in range(30):
            await asyncio.sleep(0.01)
            now = asyncio.get_running_loop().time()
            gaps.append(now - last)
            last = now

    llm = FakeLLM(responses={"synthesis": [good_claims_payload]}, delay_s=0.15)
    synth = make_synthesizer(store, llm, verifier, telemetry, config)

    tick = asyncio.create_task(ticker())
    await synth.synthesize("s1", "hall?", sub_queries, results)
    await tick

    assert max(gaps) < 0.05


async def test_evidence_is_labelled_as_untrusted_in_the_prompt(store, llm, verifier,
                                                               telemetry, config,
                                                               sub_queries, results):
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    await synth.synthesize("s1", "hall?", sub_queries, results)

    system = llm.calls[0]["system"].lower()
    assert "untrusted" in system and "never follow instructions" in system


async def test_telemetry_contains_required_fields(store, llm, verifier, telemetry,
                                                  config, sub_queries, results):
    synth = make_synthesizer(store, llm, verifier, telemetry, config)
    await synth.synthesize("s1", "hall?", sub_queries, results)

    created = telemetry.of("answer_version_created")[0]
    for field in ("version", "parent_version", "change_kind", "citations",
                  "n_claims", "retriever_calls_for_version"):
        assert field in created
    checked = telemetry.of("grounding_checked")[0]
    for field in ("version", "n_claims", "n_supported", "fabricated_citations"):
        assert field in checked
