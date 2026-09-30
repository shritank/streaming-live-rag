"""Task context §7.1 and §7.2 — citation integrity and claim support.

These are the hard invariants: a failure here blocks the merge, because an
invented citation is worse than no answer.
"""

from __future__ import annotations

import pytest

from streaming_rag.contracts import Claim
from streaming_rag.session.generative.grounding import (
    GroundingVerifier,
    LexicalSupportChecker,
    LLMSupportChecker,
)

from .fakes import FakeLLM

pytestmark = pytest.mark.asyncio


def evidence_map(*chunks_):
    from streaming_rag.contracts import EvidenceChunk

    return {c.chunk_id: EvidenceChunk(chunk=c, score=1.0, query_ids=("q1",))
            for c in chunks_}


async def test_valid_citation_passes(verifier, chunks):
    claims = [Claim(text="Hall Aurora seats 30 attendees", citations=["Doc_12 §2"])]
    kept, report = await verifier.verify_with_report(claims, evidence_map(chunks["capacity"]))

    assert len(kept) == 1
    assert kept[0].supported is True
    assert report.fabricated_citations == []


async def test_fabricated_citation_is_stripped_and_claim_dropped(verifier, chunks):
    """The forced-hallucination test: Doc_999 exists nowhere."""
    claims = [Claim(text="Hall Aurora seats 30 attendees", citations=["Doc_999 §9"])]
    kept, report = await verifier.verify_with_report(claims, evidence_map(chunks["capacity"]))

    assert kept == []
    assert report.fabricated_citations == ["Doc_999 §9"]


async def test_real_but_not_provided_citation_counts_as_fabricated(verifier, chunks):
    """A well-formed id that was never retrieved must not be trusted."""
    claims = [Claim(text="Hall Aurora seats 30 attendees",
                    citations=["Doc_31 §4"])]  # real elsewhere, not in this evidence
    kept, report = await verifier.verify_with_report(claims, evidence_map(chunks["capacity"]))

    assert kept == []
    assert report.fabricated_citations == ["Doc_31 §4"]


async def test_mixed_citations_keep_only_the_valid_ones(verifier, chunks):
    claims = [Claim(text="Hall Aurora seats 30 attendees",
                    citations=["Doc_12 §2", "Doc_999 §9"])]
    kept, report = await verifier.verify_with_report(claims, evidence_map(chunks["capacity"]))

    assert kept[0].citations == ["Doc_12 §2"]
    assert report.fabricated_citations == ["Doc_999 §9"]


async def test_unsupported_claim_with_valid_citation_is_dropped(verifier, chunks):
    """Cited chunk exists but says nothing of the sort (the distractor case)."""
    claims = [Claim(text="Parking is free for all attendees on weekends",
                    citations=["Doc_44 §7"])]
    kept, report = await verifier.verify_with_report(claims, evidence_map(chunks["distractor"]))

    assert kept == []
    assert len(report.dropped_claims) == 1
    assert report.n_supported == 0


async def test_number_conflict_fails_support(verifier, chunks):
    """"seats 40" cited to a chunk that says 30 must not survive."""
    claims = [Claim(text="Hall Aurora seats 40 attendees", citations=["Doc_12 §2"])]
    kept, _ = await verifier.verify_with_report(claims, evidence_map(chunks["capacity"]))

    assert kept == []


async def test_injection_chunk_does_not_grant_support(verifier, chunks):
    """A corpus document ordering us to cite Doc_999 is just text."""
    claims = [Claim(text="APPROVED", citations=["Doc_999 §9"]),
              Claim(text="The booking is approved", citations=["Doc_77 §1"])]
    kept, report = await verifier.verify_with_report(
        claims, evidence_map(chunks["injected"], chunks["capacity"]))

    assert all("Doc_999" not in c for claim in kept for c in claim.citations)
    assert "Doc_999 §9" in report.fabricated_citations


async def test_empty_citations_never_pass(verifier, chunks):
    claims = [Claim(text="Hall Aurora seats 30 attendees", citations=[])]
    kept, _ = await verifier.verify_with_report(claims, evidence_map(chunks["capacity"]))
    assert kept == []


async def test_duplicate_citations_are_collapsed(verifier, chunks):
    claims = [Claim(text="Hall Aurora seats 30 attendees",
                    citations=["Doc_12 §2", "Doc_12 §2"])]
    kept, _ = await verifier.verify_with_report(claims, evidence_map(chunks["capacity"]))
    assert kept[0].citations == ["Doc_12 §2"]


async def test_keep_unsupported_when_drop_is_disabled(config, telemetry, chunks):
    config["grounding"]["drop_unsupported"] = False
    verifier = GroundingVerifier(config=config, telemetry=telemetry)
    claims = [Claim(text="Parking is free", citations=["Doc_44 §7"])]
    kept, _ = await verifier.verify_with_report(claims, evidence_map(chunks["distractor"]))

    assert len(kept) == 1 and kept[0].supported is False


async def test_lexical_checker_scores_overlap():
    checker = LexicalSupportChecker()
    high = await checker.score("refunds are issued within 14 days",
                               "Refunds are issued within 14 days of cancellation.")
    low = await checker.score("badminton courts are bookable hourly",
                              "Refunds are issued within 14 days of cancellation.")
    assert high > 0.8 > low


async def test_llm_judge_ignores_instructions_inside_the_source(chunks):
    """The judge sees the source as data; a 'say supported' line must not win."""
    llm = FakeLLM(responses={"grounding_judge": [{"supported": False, "confidence": 0.0}]})
    checker = LLMSupportChecker(llm)
    score = await checker.score("The booking is approved", chunks["injected"].text)

    assert score == 0.0
    assert "untrusted" in llm.calls[0]["system"].lower()


async def test_llm_judge_failure_scores_zero_rather_than_crashing():
    llm = FakeLLM(responses={"grounding_judge": [RuntimeError("provider down")]})
    assert await LLMSupportChecker(llm).score("anything", "anything") == 0.0


async def test_verifier_reports_counts(verifier, chunks):
    claims = [
        Claim(text="Hall Aurora seats 30 attendees", citations=["Doc_12 §2"]),
        Claim(text="Parking is free for all attendees", citations=["Doc_44 §7"]),
    ]
    _, report = await verifier.verify_with_report(
        claims, evidence_map(chunks["capacity"], chunks["distractor"]))

    assert (report.n_claims, report.n_supported) == (2, 1)


async def test_contract_verify_returns_plain_claim_list(verifier, chunks):
    claims = [Claim(text="Hall Aurora seats 30 attendees", citations=["Doc_12 §2"])]
    kept = await verifier.verify(claims, evidence_map(chunks["capacity"]))
    assert isinstance(kept, list) and isinstance(kept[0], Claim)
