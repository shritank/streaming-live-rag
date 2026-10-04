"""controller.mode="model": the learned stability classifier (controller/model_stability.py)."""
from __future__ import annotations

import pytest

from streaming_rag.config import load_config
from streaming_rag.contracts import Decision, TranscriptChunk
from streaming_rag.controller import RetrievalController, model_stability
from streaming_rag.session.store import SessionStore


def _model_config():
    cfg = load_config()
    cfg.controller.mode = "model"
    return cfg


@pytest.fixture
def session():
    store = SessionStore()
    store.open("s1")
    return store


def test_feature_vector_matches_the_declared_feature_names():
    x = model_stability.features("What is the cancellation policy for Pune?", "")
    assert len(x) == len(model_stability.FEATURES)
    assert all(isinstance(v, float) and 0.0 <= v <= 2.0 for v in x)


def test_score_is_one_at_the_end_zero_when_empty_and_a_probability_otherwise():
    assert model_stability.score("anything", "", True) == 1.0
    assert model_stability.score("   ", "", False) == 0.0
    assert 0.0 < model_stability.score("What is the cancellation policy", "", False) < 1.0


def test_a_finished_question_is_more_ready_than_a_dangling_fragment():
    done = model_stability.score("What is the cancellation policy for the Pune venue?", "", False)
    cut = model_stability.score("What is the cancellation policy for the", "", False)
    assert done > cut


def test_weights_file_matches_the_feature_set():
    model = model_stability.load_model()
    assert tuple(model["features"]) == model_stability.FEATURES
    assert len(model["weights"]) == len(model_stability.FEATURES)
    assert 0.0 < model["threshold"] < 1.0


def test_an_unknown_controller_mode_is_still_refused_at_build_time():
    from streaming_rag.build import build_real_components
    cfg = load_config()
    cfg.controller.mode = "hybrid"
    with pytest.raises(ValueError, match="not implemented"):
        build_real_components(cfg)


async def test_model_controller_retrieves_on_a_finished_question_and_waits_on_a_fragment(session):
    controller = RetrievalController(None, None, _model_config())
    wait = await controller.on_chunk(TranscriptChunk("s1", "u1", 0, "What is the cancellation policy for the", False),
                                     session.view("s1"))
    assert wait.decision == Decision.WAIT
    controller = RetrievalController(None, None, _model_config())
    go = await controller.on_chunk(TranscriptChunk("s1", "u2", 0, "What is the cancellation policy for the Pune venue?", False),
                                   session.view("s1"))
    assert go.decision == Decision.RETRIEVE and go.sub_queries


async def test_model_controller_always_retrieves_at_the_end_of_the_utterance(session):
    controller = RetrievalController(None, None, _model_config())
    final = await controller.on_chunk(TranscriptChunk("s1", "u1", 0, "I need a workshop in Pune for 30 people", True),
                                      session.view("s1"))
    assert final.decision == Decision.RETRIEVE
