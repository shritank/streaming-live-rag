"""Ground-truth firewall (Task 4 §4.5): the engine must never see
ground_truth or any `_`-prefixed key, and streaming_rag/ must never import
from eval/.
"""
from __future__ import annotations

import ast
from pathlib import Path

from eval.loader import strip_ground_truth, load_events

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_strip_ground_truth_removes_ground_truth_and_underscore_keys():
    scenario = {
        "scenario_id": "x",
        "events": [{"timestamp_ms": 0, "event_type": "session_start",
                     "payload": {"session_id": "s1", "_secret": "nope"}}],
        "ground_truth": {"answer": "should never reach the engine"},
        "_private": "also stripped",
    }
    cleaned = strip_ground_truth(scenario)
    assert "ground_truth" not in cleaned
    assert "_private" not in cleaned
    assert "_secret" not in cleaned["events"][0]["payload"]
    assert cleaned["events"][0]["payload"]["session_id"] == "s1"


def test_load_events_never_exposes_ground_truth(tmp_path):
    import json
    scenario = {
        "scenario_id": "x",
        "events": [{"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": "s1"}}],
        "ground_truth": {"secret": "leak"},
    }
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    events = load_events(path)
    assert "ground_truth" not in str(events)


class SpyEngine:
    """Asserts no delivered event contains ground_truth or _-prefixed keys."""
    def __init__(self):
        self.seen: list[dict] = []

    async def handle(self, event: dict) -> None:
        assert "ground_truth" not in event
        for key in event.get("payload", {}):
            assert not key.startswith("_")
        self.seen.append(event)


async def test_spy_engine_receives_only_clean_events():
    import json
    scenario = {
        "scenario_id": "x",
        "events": [
            {"timestamp_ms": 0, "event_type": "session_start",
             "payload": {"session_id": "s1", "_leak": "no"}},
        ],
        "ground_truth": {"turns": {}},
    }
    cleaned_events = strip_ground_truth(scenario)["events"]
    spy = SpyEngine()
    for e in cleaned_events:
        await spy.handle(e)
    assert len(spy.seen) == 1


def test_streaming_rag_never_imports_from_eval():
    streaming_rag_dir = REPO_ROOT / "streaming_rag"
    offenders = []
    for path in streaming_rag_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if name and (name == "eval" or name.startswith("eval.")):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: imports {name}")
    assert not offenders, "streaming_rag/ must never import from eval/:\n" + "\n".join(offenders)
