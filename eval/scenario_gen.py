"""Seeded scenario generator.

    python -m eval.scenario_gen --template multi_intent --n 10 --seed 1 --out generated/

Templates read queries from a corpus's qrels.jsonl rather than any fixed
entity list, so a new --seed or a swapped --corpus-dir re-skins every
scenario (different cities, articles, numbers, phrasings) without touching
this file — which is what proves nothing is hardcoded.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from streaming_rag.controller.stream import chunk_utterance

TEMPLATES = ("multi_intent", "late_detail", "suppression", "chit_chat", "unanswerable")

# Each line must fully match controller._CHITCHAT_RE (which matches the
# whole utterance against one canonical phrase) so generated ground truth
# stays something the controller can actually be expected to detect.
_CHITCHAT_LINES = ["Hi", "Hello", "Thanks", "Thank you", "Okay", "Good morning"]
# Every lead-in must match controller._REFINEMENT_RE so the generated ground
# truth ("kind": "refinement") is something the controller can actually be
# expected to detect from the text alone, independent of topic overlap.
_REFINE_LEAD_INS = ["Actually,", "Wait, one more thing —", "Also,", "It turns out"]
_PRESENTATION_ASKS = [
    "Please repeat your last answer in two bullets.",
    "Can you shorten that to one sentence?",
    "Summarize what you just said as three bullet points.",
]
_UNANSWERABLE_TOPICS = [
    "the parking validation policy for visiting contractors",
    "the office plant watering schedule",
    "the company's stance on interstellar travel expenses",
]


def _load_qrels(corpus_dir: Path) -> list[dict]:
    path = corpus_dir / "qrels.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no qrels.jsonl in {corpus_dir}; run the corpus builder first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _events_for_utterance(text: str, utterance_id: str, session_id: str, start_ms: int) -> list[dict]:
    return chunk_utterance(text, utterance_id, session_id, start_ms=start_ms)


def _has_early_opportunity(text: str, words_per_chunk: int = 6) -> bool:
    """chunk_utterance groups `words_per_chunk` words per fragment, so an
    utterance with fewer words than that is delivered as a single chunk
    immediately followed by the empty is_final chunk — there is no earlier
    moment at which retrieval COULD have started. Marking such a turn
    'eligible_for_early_retrieval' in ground truth would penalise G2 for a
    structural impossibility, not a controller shortcoming."""
    return len(text.split()) > words_per_chunk


def _wrap_session(events: list[dict], session_id: str) -> list[dict]:
    last_ts = events[-1]["timestamp_ms"] if events else 0
    return (
        [{"timestamp_ms": 0, "event_type": "session_start", "payload": {"session_id": session_id}}]
        + events
        + [{"timestamp_ms": last_ts + 2000, "event_type": "session_end", "payload": {"session_id": session_id}}]
    )


def _doc_of(qrel: dict) -> str:
    return qrel["relevant"][0].split(" §")[0] if qrel["relevant"] else ""


def gen_multi_intent(rng: random.Random, qrels: list[dict], idx: int) -> dict:
    # A real compound request bundles sub-intents about ONE topic (the
    # guide's own example: venue capacity + cancellation policy + catering,
    # all about one workshop). Gluing fully unrelated queries from unrelated
    # source documents produces ambiguous pronoun references ("this network")
    # that no retriever could resolve correctly — that's a scenario-quality
    # bug, not a system bug, so prefer same-document groupings and only fall
    # back to a cross-document mix when a document doesn't have enough
    # distinct queries of its own.
    n = rng.choice([2, 3])
    by_doc: dict[str, list[dict]] = {}
    for q in qrels:
        by_doc.setdefault(_doc_of(q), []).append(q)
    same_doc_options = [group for group in by_doc.values() if len(group) >= n]
    if same_doc_options:
        picked = rng.sample(rng.choice(same_doc_options), n)
    else:
        picked = rng.sample(qrels, min(n, len(qrels)))
    clauses = [q["query"] for q in picked]
    text = clauses[0] + ", " + " and ".join(clauses[1:])
    events = _events_for_utterance(text, "u1", "s1", 0)
    gt = {"turns": {"u1": {
        "kind": "new_request", "retrieval_required": True,
        "eligible_for_early_retrieval": _has_early_opportunity(text),
        "gold_sub_intents": clauses,
        "expected_citation_docs": [r for q in picked for r in q["relevant"]],
    }}}
    return {"scenario_id": f"gen_multi_intent_{idx:03d}", "template": "multi_intent",
            "events": _wrap_session(events, "s1"), "ground_truth": gt}


def gen_late_detail(rng: random.Random, qrels: list[dict], idx: int) -> dict:
    # A late-arriving constraint refines the SAME topic (guide Example 2:
    # "the trip was international" narrows the open reimbursement question,
    # it doesn't switch to catering). Prefer a second query on the same
    # document so the delta is topically coherent, not just text-cued.
    q1 = rng.choice(qrels)
    same_doc = [q for q in qrels if q is not q1 and _doc_of(q) == _doc_of(q1)]
    q2 = rng.choice(same_doc) if same_doc else rng.choice([q for q in qrels if q is not q1])
    events = _events_for_utterance(q1["query"], "u1", "s1", 0)
    last_ts = events[-1]["timestamp_ms"] + 1500
    lead = rng.choice(_REFINE_LEAD_INS)
    refine_text = f"{lead} {q2['query']}"
    events2 = _events_for_utterance(refine_text, "u2", "s1", last_ts)
    gt = {"turns": {
        "u1": {"kind": "new_request", "retrieval_required": True,
               "eligible_for_early_retrieval": _has_early_opportunity(q1["query"]),
               "gold_sub_intents": [q1["query"]], "expected_citation_docs": q1["relevant"]},
        "u2": {"kind": "refinement", "retrieval_required": True, "eligible_for_early_retrieval": False,
               "gold_sub_intents": [q2["query"]], "expected_citation_docs": q2["relevant"]},
    }}
    return {"scenario_id": f"gen_late_detail_{idx:03d}", "template": "late_detail",
            "events": _wrap_session(events + events2, "s1"), "ground_truth": gt}


def gen_suppression(rng: random.Random, qrels: list[dict], idx: int) -> dict:
    q1 = rng.choice(qrels)
    events = _events_for_utterance(q1["query"], "u1", "s1", 0)
    last_ts = events[-1]["timestamp_ms"] + 1500
    ask = rng.choice(_PRESENTATION_ASKS)
    events2 = _events_for_utterance(ask, "u2", "s1", last_ts)
    gt = {"turns": {
        "u1": {"kind": "new_request", "retrieval_required": True,
               "eligible_for_early_retrieval": _has_early_opportunity(q1["query"]),
               "gold_sub_intents": [q1["query"]], "expected_citation_docs": q1["relevant"]},
        "u2": {"kind": "presentation_only", "retrieval_required": False, "eligible_for_early_retrieval": False},
    }}
    return {"scenario_id": f"gen_suppression_{idx:03d}", "template": "suppression",
            "events": _wrap_session(events + events2, "s1"), "ground_truth": gt}


def gen_chit_chat(rng: random.Random, qrels: list[dict], idx: int) -> dict:
    line = rng.choice(_CHITCHAT_LINES)
    events = _events_for_utterance(line, "u1", "s1", 0)
    gt = {"turns": {"u1": {"kind": "chit_chat", "retrieval_required": False,
                            "eligible_for_early_retrieval": False}}}
    return {"scenario_id": f"gen_chit_chat_{idx:03d}", "template": "chit_chat",
            "events": _wrap_session(events, "s1"), "ground_truth": gt}


def gen_unanswerable(rng: random.Random, qrels: list[dict], idx: int) -> dict:
    topic = rng.choice(_UNANSWERABLE_TOPICS)
    text = f"What is {topic}?"
    events = _events_for_utterance(text, "u1", "s1", 0)
    gt = {"turns": {"u1": {"kind": "new_request", "retrieval_required": True,
                            "eligible_for_early_retrieval": False, "gold_sub_intents": [text],
                            "expect_uncertainty": True}}}
    return {"scenario_id": f"gen_unanswerable_{idx:03d}", "template": "unanswerable",
            "events": _wrap_session(events, "s1"), "ground_truth": gt}


_GENERATORS = {
    "multi_intent": gen_multi_intent,
    "late_detail": gen_late_detail,
    "suppression": gen_suppression,
    "chit_chat": gen_chit_chat,
    "unanswerable": gen_unanswerable,
}


def generate(template: str, n: int, seed: int, corpus_dir: Path, out_dir: Path) -> list[Path]:
    if template not in _GENERATORS:
        raise ValueError(f"unknown template {template!r}, choose from {list(_GENERATORS)}")
    qrels = _load_qrels(corpus_dir)
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(n):
        scenario = _GENERATORS[template](rng, qrels, i)
        path = out_dir / f"{scenario['scenario_id']}.json"
        path.write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        written.append(path)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, choices=list(_GENERATORS))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--corpus-dir", default="fixtures/dev_corpus")
    parser.add_argument("--out", default="generated")
    args = parser.parse_args(argv)
    written = generate(args.template, args.n, args.seed, Path(args.corpus_dir), Path(args.out))
    print(f"wrote {len(written)} scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
