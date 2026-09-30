"""The CLI replay command the demo uses: its output shape and the opt-in --warmup pass."""
from __future__ import annotations

import json

from streaming_rag.cli import main

SCENARIO = "eval/scenarios/dev_002_late_detail.json"


def _replay(capsys, *extra):
    assert main(["replay", SCENARIO, "--impl", "mock", "--time-scale", "200", *extra]) == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_replay_prints_one_turn_result_per_utterance(capsys):
    turns = _replay(capsys)
    assert [t["utterance_id"] for t in turns] == ["u1", "u2", "u3"]
    assert [t["answer_version"] for t in turns] == [1, 2, 3]
    assert turns[2]["retrieval_events"] == [] and turns[2]["retrieval_required"] is False


def test_warmup_pass_changes_nothing_in_the_output(capsys):
    """--warmup replays once, uncounted, before the real run; the printed turns must be the real run's only."""
    plain = _replay(capsys)
    warmed = _replay(capsys, "--warmup")
    assert len(warmed) == len(plain) == 3
    assert [t["answer_version"] for t in warmed] == [t["answer_version"] for t in plain]
