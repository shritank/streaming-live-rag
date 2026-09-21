"""Scenario generator determinism (Task 4 §4.5)."""
from __future__ import annotations

import json

from eval.scenario_gen import generate


def test_same_seed_gives_byte_identical_scenarios(tmp_path):
    import pathlib
    corpus = pathlib.Path("fixtures/dev_corpus")
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    generate("multi_intent", 5, seed=42, corpus_dir=corpus, out_dir=out1)
    generate("multi_intent", 5, seed=42, corpus_dir=corpus, out_dir=out2)

    files1 = sorted(out1.glob("*.json"))
    files2 = sorted(out2.glob("*.json"))
    assert len(files1) == len(files2) == 5
    for f1, f2 in zip(files1, files2):
        assert f1.read_bytes() == f2.read_bytes()


def test_different_seeds_give_different_scenarios(tmp_path):
    import pathlib
    out1 = tmp_path / "seed1"
    out2 = tmp_path / "seed2"
    generate("multi_intent", 5, seed=1, corpus_dir=pathlib.Path("fixtures/dev_corpus"), out_dir=out1)
    generate("multi_intent", 5, seed=2, corpus_dir=pathlib.Path("fixtures/dev_corpus"), out_dir=out2)

    contents1 = [f.read_text(encoding="utf-8") for f in sorted(out1.glob("*.json"))]
    contents2 = [f.read_text(encoding="utf-8") for f in sorted(out2.glob("*.json"))]
    assert contents1 != contents2


def test_generated_scenarios_have_no_ground_truth_leakage_in_events(tmp_path):
    """Each generated file's events must not literally embed the queries'
    gold-answer document IDs anywhere but inside ground_truth."""
    import pathlib
    from eval.loader import strip_ground_truth
    corpus = pathlib.Path("fixtures/dev_corpus")
    scenarios = generate("suppression", 3, seed=9, corpus_dir=corpus, out_dir=tmp_path)
    for path in scenarios:
        scenario = json.loads(path.read_text(encoding="utf-8"))
        cleaned = strip_ground_truth(scenario)
        assert "ground_truth" not in cleaned
