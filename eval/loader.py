"""Scenario loading with the ground-truth firewall.

`ground_truth` and any `_`-prefixed key are stripped before the engine ever
sees the scenario (project context §9.9). This module is the one and only
place that reads `ground_truth`; `streaming_rag/` must never import from
`eval/` (enforced by tests/eval/test_firewall.py).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def strip_ground_truth(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_ground_truth(v) for k, v in obj.items()
                 if k != "ground_truth" and not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_ground_truth(v) for v in obj]
    return obj


def load_scenario_file(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_events(path: str | Path) -> list[dict]:
    """What the engine is allowed to see: events only, ground_truth stripped."""
    scenario = load_scenario_file(path)
    return strip_ground_truth(scenario)["events"]


def load_ground_truth(path: str | Path) -> dict:
    """What the evaluator (never the engine) is allowed to see."""
    scenario = load_scenario_file(path)
    return scenario.get("ground_truth", {})


def discover_scenarios(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("**/*.json"))
