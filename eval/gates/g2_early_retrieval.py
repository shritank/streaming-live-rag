"""G2 — early retrieval: >= 80% of eligible queries retrieve before the
utterance ends; false-trigger rate on no-retrieval cases <= 10%.

Ground truth schema (per scenario, under "turns"):
    {"u1": {"retrieval_required": true, "eligible_for_early_retrieval": true}}

Eligibility means the guide's early-retrieval mechanism should have enough
signal to fire before utterance_end (e.g. multi-word requests with a named
entity partway through) — a bare "hi" is never eligible even though it also
has retrieval_required=false.
"""
from __future__ import annotations

from dataclasses import dataclass

from streaming_rag.telemetry.trace import assemble_turns


@dataclass
class G2Result:
    eligible: int
    early: int
    no_retrieval_cases: int
    false_triggers: int

    @property
    def early_rate(self) -> float:
        return self.early / self.eligible if self.eligible else 1.0

    @property
    def false_trigger_rate(self) -> float:
        return self.false_triggers / self.no_retrieval_cases if self.no_retrieval_cases else 0.0

    @property
    def passed(self) -> bool:
        return self.early_rate >= 0.80 and self.false_trigger_rate <= 0.10


def score_g2(trace_events: list[dict], ground_truth: dict) -> G2Result:
    turns_gt = ground_truth.get("turns", {})
    by_utt = {(t.session_id, t.utterance_id): t for t in assemble_turns(trace_events)}

    eligible = early = no_retrieval_cases = false_triggers = 0

    for utterance_id, gt in turns_gt.items():
        turn = next((t for (s, u), t in by_utt.items() if u == utterance_id), None)
        if turn is None:
            continue
        utterance_end = turn.first("chunk_received")
        end_ts = None
        for e in turn.of("chunk_received"):
            if e.get("is_final"):
                end_ts = e.get("ts_ms")

        first_retrieval = turn.first("retrieval_started")

        if gt.get("eligible_for_early_retrieval"):
            eligible += 1
            if first_retrieval is not None and end_ts is not None and \
               first_retrieval.get("ts_ms", float("inf")) < end_ts:
                early += 1
        elif not gt.get("retrieval_required", True):
            no_retrieval_cases += 1
            if first_retrieval is not None:
                false_triggers += 1

    return G2Result(eligible, early, no_retrieval_cases, false_triggers)
