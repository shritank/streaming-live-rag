"""Gate scorer correctness (Task 4 §4.5): hand-crafted traces with known
answers, asserting each scorer returns exactly the expected value.
"""
from __future__ import annotations

from eval.gates import score_g2, score_g3, score_g4, score_g5


def _turn(session_id, utterance_id, events):
    for e in events:
        e.setdefault("session_id", session_id)
        e.setdefault("utterance_id", utterance_id)
    return events


def test_g2_eight_of_ten_early_retrievals_scores_80_percent():
    trace = []
    ground_truth = {"turns": {}}
    for i in range(10):
        uid = f"u{i}"
        early = i < 8
        ground_truth["turns"][uid] = {"eligible_for_early_retrieval": True}
        trace += _turn("s1", uid, [
            {"event": "chunk_received", "ts_ms": 0, "is_final": False, "text_len": 5},
            {"event": "chunk_received", "ts_ms": 1000, "is_final": True, "text_len": 0},
            {"event": "retrieval_started", "ts_ms": 500 if early else 1200, "request_id": f"r{i}",
             "query_ids": [f"{uid}.q1"], "mode": "hybrid"},
        ])
    result = score_g2(trace, ground_truth)
    assert result.eligible == 10
    assert result.early == 8
    assert result.early_rate == 0.8


def test_g2_false_trigger_on_no_retrieval_case():
    ground_truth = {"turns": {"u1": {"eligible_for_early_retrieval": False, "retrieval_required": False}}}
    trace = _turn("s1", "u1", [
        {"event": "chunk_received", "ts_ms": 0, "is_final": True, "text_len": 2},
        {"event": "retrieval_started", "ts_ms": 0, "request_id": "r1", "query_ids": ["q1"], "mode": "hybrid"},
    ])
    result = score_g2(trace, ground_truth)
    assert result.no_retrieval_cases == 1
    assert result.false_triggers == 1
    assert result.false_trigger_rate == 1.0


def test_g3_compound_query_split_into_two_or_more():
    ground_truth = {"turns": {
        "u1": {"gold_sub_intents": ["a thing", "another thing"]},
        "u2": {"gold_sub_intents": ["only one thing"]},  # not compound, excluded
    }}
    trace = (
        _turn("s1", "u1", [{"event": "sub_queries_emitted", "ts_ms": 0, "query_ids": ["q1", "q2"],
                              "queries": ["a", "b"], "trigger": "multi_intent"}])
        + _turn("s1", "u2", [{"event": "sub_queries_emitted", "ts_ms": 0, "query_ids": ["q3"],
                                "queries": ["c"], "trigger": "provisional"}])
    )
    result = score_g3(trace, ground_truth)
    assert result.compound == 1
    assert result.correctly_split == 1
    assert result.rate == 1.0


def test_g4_fabricated_citation_is_counted():
    trace = [
        {"event": "grounding_checked", "session_id": "s1", "utterance_id": "u1", "ts_ms": 0,
         "n_claims": 2, "n_supported": 1, "fabricated_citations": ["Doc_999 §1"]},
    ]
    result = score_g4(trace, {})
    assert result.n_claims == 2
    assert result.n_supported == 1
    assert result.fabricated == ["Doc_999 §1"]
    assert not result.passed  # fabrication alone fails the gate regardless of support_rate


def test_g4_passes_with_zero_fabrication_and_high_support():
    trace = [
        {"event": "grounding_checked", "session_id": "s1", "utterance_id": "u1", "ts_ms": 0,
         "n_claims": 10, "n_supported": 9, "fabricated_citations": []},
    ]
    result = score_g4(trace, {})
    assert result.support_rate == 0.9
    assert result.passed


def test_g5_refinement_that_reruns_v1_queries_fails():
    ground_truth = {"turns": {"u2": {"kind": "refinement"}}}
    trace = (
        _turn("s1", "u1", [
            {"event": "answer_version_created", "ts_ms": 0, "version": 1, "parent_version": None,
             "change_kind": "initial", "citations": ["Doc_01 §1"]},
        ])
        + _turn("s1", "u2", [
            # a "refinement" that restarts from scratch: change_kind is "initial", not "refinement"
            {"event": "answer_version_created", "ts_ms": 1000, "version": 2, "parent_version": None,
             "change_kind": "initial", "citations": ["Doc_01 §1"]},
        ])
    )
    result = score_g5(trace, ground_truth)
    assert result.refinement_turns == 1
    assert result.continuous == 0
    assert not result.passed
    assert any("expected 'refinement'" in f for f in result.failures)


def test_g5_refinement_that_drops_prior_citations_fails():
    ground_truth = {"turns": {"u2": {"kind": "refinement"}}}
    trace = (
        _turn("s1", "u1", [
            {"event": "answer_version_created", "ts_ms": 0, "version": 1, "parent_version": None,
             "change_kind": "initial", "citations": ["Doc_01 §1", "Doc_02 §1"]},
        ])
        + _turn("s1", "u2", [
            {"event": "answer_version_created", "ts_ms": 1000, "version": 2, "parent_version": 1,
             "change_kind": "refinement", "citations": ["Doc_03 §1"]},  # dropped Doc_01, Doc_02
        ])
    )
    result = score_g5(trace, ground_truth)
    assert result.continuous == 0
    assert any("dropped prior citations" in f for f in result.failures)


def test_g5_proper_refinement_passes():
    ground_truth = {"turns": {"u2": {"kind": "refinement"}}}
    trace = (
        _turn("s1", "u1", [
            {"event": "answer_version_created", "ts_ms": 0, "version": 1, "parent_version": None,
             "change_kind": "initial", "citations": ["Doc_01 §1"]},
        ])
        + _turn("s1", "u2", [
            {"event": "answer_version_created", "ts_ms": 1000, "version": 2, "parent_version": 1,
             "change_kind": "refinement", "citations": ["Doc_01 §1", "Doc_02 §1"]},
        ])
    )
    result = score_g5(trace, ground_truth)
    assert result.continuous == 1
    assert result.passed
