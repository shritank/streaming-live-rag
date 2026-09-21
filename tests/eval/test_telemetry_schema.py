"""G6 schema validation and coverage checker (Task 4 §4.3)."""
from __future__ import annotations

import asyncio

from eval.gates.g6_coverage import score_g6
from eval.schema_check import load_schema, validate_events
from tests.engine.conftest import run_events
from tests.engine.test_reference_patterns import multi_intent_events


async def test_full_scenario_trace_validates_against_schema():
    _, trace = await run_events(multi_intent_events())
    errors = validate_events(trace, load_schema())
    assert errors == []


async def test_full_scenario_trace_has_100_percent_g6_coverage():
    _, trace = await run_events(multi_intent_events())
    result = score_g6(trace)
    assert result.total_turns == 1
    assert result.covered_turns == 1
    assert result.passed


def test_g6_flags_orphaned_retrieval_completed():
    trace = [
        {"event": "chunk_received", "session_id": "s1", "utterance_id": "u1", "ts_ms": 0,
         "is_final": True, "text_len": 3},
        {"event": "controller_decision", "session_id": "s1", "utterance_id": "u1", "ts_ms": 0,
         "decision": "retrieve", "turn_kind": "new_request", "reason": "x", "stability": 1.0,
         "n_sub_queries": 1, "decision_latency_ms": 1.0},
        # retrieval_completed with no matching retrieval_started
        {"event": "retrieval_completed", "session_id": "s1", "utterance_id": "u1", "ts_ms": 10,
         "request_id": "orphan", "latency_ms": 1.0, "chunk_ids": [], "low_confidence_query_ids": []},
    ]
    result = score_g6(trace)
    assert result.total_turns == 1
    assert result.covered_turns == 0
    assert any("no matching retrieval_started" in p for p in result.problems)


def test_g6_flags_missing_controller_decision():
    trace = [
        {"event": "chunk_received", "session_id": "s1", "utterance_id": "u1", "ts_ms": 0,
         "is_final": True, "text_len": 3},
    ]
    result = score_g6(trace)
    assert result.covered_turns == 0
    assert any("missing controller_decision" in p for p in result.problems)
