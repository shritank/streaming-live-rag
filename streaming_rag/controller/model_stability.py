"""Learned stability scorer for `controller.mode="model"` (the guide's "rule-based vs model-based
controller" ablation).

The rule scorer (stability.score) adds up hand-set weights over a few structural signals. This one
learns the combination: a logistic regression over the same signals (anchors, mid-thought ending, "nothing
new arrived") plus length and question-shape features. It is a classifier, not an LLM: it adds no tokens,
no network call and ~microseconds per chunk, so it keeps the "fast rule path" parsimony of the design.

Weights live in model_controller.json and are produced by `python -m eval.train_model_controller`, which
trains on DEV scenarios only (label: "enough of a gold sub-question has been spoken to search on").
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..retrieval.text import content_tokens
from . import stability

WEIGHTS_PATH = Path(__file__).with_name("model_controller.json")
_WH = frozenset("what who whom whose when where why how which".split())

FEATURES = ("strong_anchor", "topic_anchor", "bigram_anchor", "weak_anchor", "ends_mid_thought",
            "no_new_content", "n_content", "n_words", "wh_opener", "ends_question")


def features(text: str, previous_text: str, known_terms=None, low_df_bigrams=None) -> list[float]:
    t = text.strip()
    tokens = content_tokens(t)
    words = t.split()
    first = [w.strip("?,.!;:").lower() for w in words[:3]]
    new = set(tokens) - set(content_tokens(previous_text))
    return [
        float(stability.has_strong_anchor(t)),
        float(stability.has_topic_anchor(t, known_terms)),
        float(stability.has_bigram_anchor(t, low_df_bigrams)),
        float(stability.has_weak_anchor(t)),
        float(stability.ends_mid_thought(t)),
        float(bool(previous_text) and not new),
        min(len(tokens) / 8.0, 2.0),
        min(len(words) / 20.0, 2.0),
        float(any(w in _WH for w in first)),
        float(t.endswith("?")),
    ]


_MODEL: dict | None = None


def load_model(path: Path | None = None) -> dict:
    global _MODEL
    if path is not None or _MODEL is None:
        p = path or WEIGHTS_PATH
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing. Train it with: python -m eval.train_model_controller")
        model = json.loads(p.read_text(encoding="utf-8"))
        if tuple(model["features"]) != FEATURES:
            raise ValueError("model_controller.json was trained with a different feature set; retrain it")
        if path is not None:
            return model
        _MODEL = model
    return _MODEL


def probability(text: str, previous_text: str, known_terms=None, low_df_bigrams=None, model: dict | None = None) -> float:
    m = model or load_model()
    x = features(text, previous_text, known_terms, low_df_bigrams)
    z = m["bias"] + sum(w * v for w, v in zip(m["weights"], x, strict=True))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def score(accumulated_text: str, previous_text: str, is_final: bool, known_terms=None, low_df_bigrams=None) -> float:
    """Drop-in for stability.score: 1.0 at the end of the utterance, else the learned P(ready to search)."""
    if is_final:
        return 1.0
    if not accumulated_text.strip():
        return 0.0
    return probability(accumulated_text, previous_text, known_terms, low_df_bigrams)
