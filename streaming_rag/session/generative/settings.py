"""Task 3's tunables, and the adapter onto Task 4's `Config`.

Task 4 owns `streaming_rag/config.py`, whose values are frozen dataclasses. The
session package keeps its own thresholds as a plain dict (there are many, and
they are calibrated rather than chosen), so this module bridges the two:
`from_engine_config()` maps the engine's Config onto these keys.
"""

from __future__ import annotations

import copy
from typing import Any

DEFAULTS: dict[str, Any] = {
    "llm": {
        # The LLM client is built by the engine (streaming_rag/llm.py) from its
        # own config and passed in; these keys only describe it for logging.
        "provider": "none",
        "model": "",
        "timeout_s": 20.0,
        "retries": 1,
        "temperature": 0.0,
        "seed": 7,
        "price_per_mtok": {"in": 0.0, "out": 0.0},
    },
    "synthesis": {
        "max_claims": 8,
        "max_evidence_chunks": 12,
        "max_chunk_chars": 900,
        "stream_partials": True,
        "citation_format": "[{citation}]",
    },
    "grounding": {
        # "lexical" needs no model download; "nli" uses a local cross-encoder;
        # "llm" asks the language model. "nli" falls back to "lexical" if the
        # model cannot be loaded.
        # Set from Config.synthesis.grounding_checker by adapter.py, not from
        # the environment: one configuration source for the whole engine.
        "checker": "lexical",
        "nli_model": "cross-encoder/nli-deberta-v3-small",
        "support_threshold": 0.55,
        "nli_threshold": 0.5,
        "number_conflict_fails": True,
        "drop_unsupported": True,
    },
    "session": {
        "max_versions": 50,
        "max_turns": 50,
    },
}


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a deep copy of the defaults with ``overrides`` merged in."""
    cfg = copy.deepcopy(DEFAULTS)
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def from_engine_config(engine_config: Any,
                       overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build session settings from Task 4's `Config` object.

    Only the keys the engine actually owns are taken across; the grounding
    thresholds stay here because they are Task 3's to calibrate.
    """
    cfg = load_config()
    llm = getattr(engine_config, "llm", None)
    if llm is not None:
        cfg["llm"].update({
            "provider": getattr(llm, "provider", cfg["llm"]["provider"]),
            "model": getattr(llm, "model", cfg["llm"]["model"]),
            "timeout_s": getattr(llm, "timeout_s", cfg["llm"]["timeout_s"]),
            "retries": getattr(llm, "max_retries", cfg["llm"]["retries"]),
            "temperature": getattr(llm, "temperature", cfg["llm"]["temperature"]),
            "seed": getattr(llm, "seed", cfg["llm"]["seed"]),
        })
    synthesis = getattr(engine_config, "synthesis", None)
    if synthesis is not None and not getattr(synthesis, "grounding_check", True):
        # The engine can switch verification off for an ablation run; claims are
        # still ID-checked, they are simply no longer dropped on weak support.
        cfg["grounding"]["drop_unsupported"] = False
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted path, e.g. ``get(cfg, "grounding.support_threshold")``."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
