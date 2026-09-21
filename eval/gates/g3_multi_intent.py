"""G3 — multi-intent identification: >= 70% of compound queries correctly
isolate at least 2 distinct gold sub-intents.

Ground truth: {"u1": {"gold_sub_intents": ["...", "...", "..."]}} — a turn
with >= 2 gold sub-intents is "compound". We check that the engine emitted
at least 2 sub-queries for that utterance (n_sub_queries from
controller_decision / sub_queries_emitted events), which is a reasonable
proxy for "isolated the distinct intents" without requiring the scorer to
know the corpus's paraphrase vocabulary (that would make this gate
content-specific, which the guide's anti-hardcoding rule forbids).
"""
from __future__ import annotations

from dataclasses import dataclass

from streaming_rag.telemetry.trace import assemble_turns


@dataclass
class G3Result:
    compound: int
    correctly_split: int

    @property
    def rate(self) -> float:
        return self.correctly_split / self.compound if self.compound else 1.0

    @property
    def passed(self) -> bool:
        return self.rate >= 0.70


def score_g3(trace_events: list[dict], ground_truth: dict) -> G3Result:
    turns_gt = ground_truth.get("turns", {})
    turns = assemble_turns(trace_events)

    compound = correctly_split = 0
    for utterance_id, gt in turns_gt.items():
        gold = gt.get("gold_sub_intents", [])
        if len(gold) < 2:
            continue
        compound += 1
        turn = next((t for t in turns if t.utterance_id == utterance_id), None)
        if turn is None:
            continue
        n_sub_queries = len({qid for e in turn.of("sub_queries_emitted") for qid in e.get("query_ids", [])})
        if n_sub_queries >= 2:
            correctly_split += 1

    return G3Result(compound, correctly_split)
