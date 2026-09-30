"""Contract conformance (task context §7.9).

Task 4 ships the canonical suite; this is the local stand-in so Task 3 can prove
conformance before the merge. It checks the *shapes* crossing the boundary, not
answer quality: Task 4's engine will call these methods and read these fields.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from streaming_rag import contracts
from streaming_rag.session.generative.grounding import GroundingVerifier
from streaming_rag.session.generative.settings import load_config
from streaming_rag.session.generative.store import SessionStore
from streaming_rag.session.generative.synthesizer import GroundedSynthesizer
from tests.session.generative.fakes import FakeLLM, chunk, claim_payload, result, sub_query


@pytest.fixture
def wired():
    config = load_config({"grounding": {"checker": "lexical", "support_threshold": 0.5}})
    store = SessionStore(config=config)
    store.open("s1")
    llm = FakeLLM(responses={"synthesis": [claim_payload([
        ("Hall Aurora seats 30 attendees", ["Doc_12 §2"], "u1.q1")])],
        "restructure": [{"text": "Short. [Doc_12 §2]"}],
        "delta_decisions": [{"decisions": [{"index": 0, "action": "keep"}]}],
        "refinement_claims": [claim_payload([])],
        "clarification": [{"question": "Which hall?"}]})
    synth = GroundedSynthesizer(
        store=store, llm=llm,
        verifier=GroundingVerifier(config=config), config=config)
    return store, synth, llm


def sample_inputs():
    c = chunk("Doc_12", "2", "Hall Aurora seats 30 attendees for seated sessions.")
    return ([sub_query("u1.q1", "hall capacity", "venue capacity")],
            [result("u1.q1", [c])])


def test_synthesizer_satisfies_the_protocol(wired):
    _, synth, _ = wired
    for name in ("synthesize", "refine", "restructure"):
        assert inspect.iscoroutinefunction(getattr(synth, name))

    expected = list(inspect.signature(contracts.Synthesizer.synthesize).parameters)
    actual = list(inspect.signature(type(synth).synthesize).parameters)
    # Extra parameters are allowed only if optional: the engine may pass
    # `on_partial`, but every contract parameter must still be accepted in order.
    assert actual[:len(expected)] == expected
    extras = inspect.signature(type(synth).synthesize).parameters
    for name in actual[len(expected):]:
        assert extras[name].default is not inspect.Parameter.empty


def test_session_view_satisfies_the_protocol(wired):
    store, _, _ = wired
    view = store.view("s1")
    assert isinstance(view.session_id, str)
    for name in ("has_answer", "latest_answer", "topic_summary", "prior_sub_queries"):
        assert callable(getattr(view, name))
    assert view.has_answer() is False
    assert view.latest_answer() is None
    assert view.topic_summary() == ""
    assert view.prior_sub_queries() == []


def test_store_satisfies_the_documented_store_api(wired):
    store, _, _ = wired
    for name in ("open", "view", "add_evidence", "close"):
        assert callable(getattr(store, name))


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_returns_real_contract_dataclasses_not_lookalikes(wired):
    _, synth, _ = wired
    queries, results = sample_inputs()
    version = await synth.synthesize("s1", "capacity?", queries, results)

    assert type(version) is contracts.AnswerVersion
    assert all(type(c) is contracts.Claim for c in version.claims)
    assert all(type(sq) is contracts.SubQuery for sq in version.sub_queries)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_answer_version_fields_are_well_typed(wired):
    _, synth, _ = wired
    queries, results = sample_inputs()
    version = await synth.synthesize("s1", "capacity?", queries, results)

    assert isinstance(version.version, int) and version.version >= 1
    assert version.parent_version is None or isinstance(version.parent_version, int)
    assert version.change_kind in ("initial", "refinement", "restructure", "clarification")
    assert isinstance(version.text, str)
    assert isinstance(version.citations, list)
    assert isinstance(version.evidence_ids, list)
    assert version.uncertainty is None or isinstance(version.uncertainty, str)
    assert isinstance(version.created_ms, int)
    # every field the contract declares is present
    declared = {f.name for f in dataclasses.fields(contracts.AnswerVersion)}
    assert declared <= set(version.__dict__)


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_version_lineage_across_all_three_modes(wired):
    _, synth, _ = wired
    queries, results = sample_inputs()

    v1 = await synth.synthesize("s1", "capacity?", queries, results)
    v2 = await synth.refine("s1", "make it larger",
                            [sub_query("u2.q1", "larger hall", "venue capacity",
                                       trigger="refinement", utterance="u2",
                                       parent="u1.q1")], results)
    v3 = await synth.restructure("s1", "shorter")

    assert [v.version for v in (v1, v2, v3)] == [1, 2, 3]
    assert [v.parent_version for v in (v1, v2, v3)] == [None, 1, 2]


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_citations_are_a_subset_of_the_evidence_given(wired):
    store, synth, _ = wired
    queries, results = sample_inputs()
    version = await synth.synthesize("s1", "capacity?", queries, results)

    available = {ev.chunk.citation for ev in store.evidence_for("s1").values()}
    assert set(version.citations) <= available


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_claims_carry_only_verified_citations(wired):
    _, synth, _ = wired
    queries, results = sample_inputs()
    version = await synth.synthesize("s1", "capacity?", queries, results)

    for claim in version.claims:
        assert claim.supported is True
        assert claim.citations


def test_synthesizer_cannot_retrieve(wired):
    """Structural guarantee behind 'restructure performs zero retrieval'."""
    _, synth, _ = wired
    assert not any("retriev" in name.lower() for name in vars(synth))


def test_null_telemetry_is_accepted(wired):
    store, _, llm = wired
    config = load_config()
    synth = GroundedSynthesizer(store=store, llm=llm, config=config,
                                telemetry=contracts.NullTelemetry())
    assert synth.telemetry is not None
