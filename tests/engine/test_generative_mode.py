"""synthesis.mode=generative through C's real engine, controller and retriever.

The generative synthesizer (from build A) and the engine (from build C) each
pass their own unit tests. These tests exist because the previous merge of A
into another engine showed that is not enough: both halves were correct and the
combined system still scored 0 on a gate. Everything here is real except the
LLM, which is an offline double (see offline_llm.py).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from eval.gates.g6_coverage import score_g6
from eval.schema_check import load_schema, validate_events
from streaming_rag.build import build_real_components
from streaming_rag.cli import load_scenario_events
from streaming_rag.config import load_config
from streaming_rag.engine import Engine
from streaming_rag.llm import build_llm_client
from streaming_rag.session.generative import GenerativeSynthesizer
from streaming_rag.session.store import SessionStore
from streaming_rag.telemetry.sinks import BufferedTelemetry
from tests.session.generative.offline_llm import OfflineClaimsLLM

ROOT = Path(__file__).resolve().parents[2]
LATE_DETAIL = ROOT / "eval" / "scenarios" / "dev_002_late_detail.json"


async def _run(scenario: Path, llm) -> tuple[list[dict], list[dict], SessionStore]:
    config = load_config()
    config.synthesis.mode = "generative"
    telemetry = BufferedTelemetry()
    store = SessionStore()
    controller, retriever, _, _, store = build_real_components(config, telemetry, store)
    synthesizer = GenerativeSynthesizer(retriever, store, config, llm=llm, telemetry=telemetry)
    engine = Engine(controller, retriever, synthesizer, llm, config, telemetry=telemetry,
                    session_store=store)

    events = await load_scenario_events(str(scenario))
    in_q: asyncio.Queue = asyncio.Queue()
    out_q: asyncio.Queue = asyncio.Queue()
    for event in events:
        in_q.put_nowait(event)
    in_q.put_nowait(None)
    await asyncio.wait_for(engine.run(in_q, out_q), timeout=120)

    outputs = []
    while not out_q.empty():
        item = out_q.get_nowait()
        if item is not None:
            outputs.append(item)
    return outputs, telemetry.events, store


def _turns(outputs: list[dict]) -> list[dict]:
    return [o for o in outputs if o.get("kind") == "turn_result"]


@pytest.mark.asyncio
async def test_late_detail_scenario_refines_instead_of_restarting():
    """The guide's Example 2, end to end: V1 → V2 refinement → V3 restructure."""
    llm = OfflineClaimsLLM()
    outputs, _, _ = await _run(LATE_DETAIL, llm)
    turns = _turns(outputs)

    assert [t["answer_version"] for t in turns] == [1, 2, 3], turns
    assert [t["parent_version"] for t in turns] == [None, 1, 2]
    assert turns[0]["citations"], "the first answer cited nothing"
    # refinement keeps what it had and adds the delta
    assert set(turns[0]["citations"]) <= set(turns[1]["citations"])
    assert len(turns[1]["citations"]) > len(turns[0]["citations"])
    # restructure invents no source and triggers no generation of new claims
    assert set(turns[2]["citations"]) <= set(turns[1]["citations"])
    assert turns[2]["retrieval_required"] is False
    assert "delta_decisions" in llm.purposes()
    assert "restructure" in llm.purposes()


@pytest.mark.asyncio
async def test_every_cited_source_exists_in_the_corpus():
    outputs, _, _ = await _run(LATE_DETAIL, OfflineClaimsLLM())
    config = load_config()
    corpus_docs = {p.stem.split("_")[0] + "_" + p.stem.split("_")[1]
                   for p in Path(ROOT / config.corpus_dir).glob("Doc_*.md")}
    for turn in _turns(outputs):
        for citation in turn["citations"]:
            assert citation.split(" ")[0] in corpus_docs, f"{citation} is not a corpus document"


@pytest.mark.asyncio
async def test_trace_stays_valid_with_exactly_one_version_event_per_turn():
    """Both components know how to emit answer_version_created; the trace
    must carry exactly one per turn, schema-valid, with full G6 coverage."""
    outputs, trace, _ = await _run(LATE_DETAIL, OfflineClaimsLLM())

    assert validate_events(trace, load_schema()) == []
    created = [e for e in trace if e.get("event") == "answer_version_created"]
    assert len(created) == len(_turns(outputs))
    g6 = score_g6(trace)
    assert g6.passed, g6.problems


@pytest.mark.asyncio
async def test_session_end_forgets_generative_state():
    _, _, store = await _run(LATE_DETAIL, OfflineClaimsLLM())
    for session_id in ("s1",):
        answers, sub_queries, evidence = store.history(session_id)
        assert (answers, sub_queries, evidence) == ([], [], [])


@pytest.mark.asyncio
async def test_generative_mode_with_the_default_fake_llm_degrades_cleanly():
    """With no real provider, the model returns "{}": every turn should become
    an honest non-answer, never an invented citation or a crash."""
    outputs, trace, _ = await _run(LATE_DETAIL, build_llm_client(load_config()))

    turns = _turns(outputs)
    assert turns, "no turn produced a result"
    for turn in turns:
        assert turn["citations"] == []
    assert validate_events(trace, load_schema()) == []
