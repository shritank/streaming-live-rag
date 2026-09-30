"""Settings that do nothing must fail, not pass silently.

An ablation is only meaningful if each arm really runs what its label says.
"""
from __future__ import annotations

import pytest

from streaming_rag.build import build_components
from streaming_rag.config import load_config


@pytest.mark.parametrize("mode", ["model", "hybrid"])
def test_unimplemented_controller_modes_are_rejected(mode):
    """No LLM controller exists; 'model' must not quietly run the rule path."""
    config = load_config()
    config.controller.mode = mode
    with pytest.raises(ValueError, match="not implemented"):
        build_components(config)


def test_unknown_synthesis_mode_is_rejected():
    config = load_config()
    config.synthesis.mode = "abstractive"
    with pytest.raises(ValueError, match="unknown synthesis.mode"):
        build_components(config)


def test_defaults_are_the_measured_configuration():
    """The defaults are the configuration the sealed held-out results came from."""
    config = load_config()
    assert config.controller.mode == "rule"
    assert config.synthesis.mode == "extractive"
    assert config.synthesis.selector == "cross-encoder"


def test_generative_settings_come_from_the_engine_config():
    """The ported synthesizer no longer reads environment variables; its
    grounding checker must follow Config.synthesis instead."""
    from streaming_rag.session.generative.adapter import settings_from_config

    config = load_config()
    assert settings_from_config(config)["grounding"]["checker"] == "lexical"
    config.synthesis.grounding_checker = "nli"
    assert settings_from_config(config)["grounding"]["checker"] == "nli"
    config.synthesis.grounding_check = False
    assert settings_from_config(config)["grounding"]["drop_unsupported"] is False
