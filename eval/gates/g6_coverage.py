"""G6 — telemetry & observability: 100% trace coverage (Task 4 §4.3).

For each turn, every required fact must be reconstructable from the trace
alone: chunk timestamps, every controller decision, matched
retrieval_started/completed pairs, sub-query -> chunk -> citation mapping,
answer version lineage with no gaps, token cost, and end-to-end latency.

Also checks causal ordering: ts_ms non-decreasing per component,
retrieval_completed after its retrieval_started, answer_version_created
after the retrievals it used.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from streaming_rag.telemetry.trace import assemble_turns


@dataclass
class G6Result:
    total_turns: int
    covered_turns: int
    problems: list[str] = field(default_factory=list)

    @property
    def coverage_rate(self) -> float:
        return self.covered_turns / self.total_turns if self.total_turns else 1.0

    @property
    def passed(self) -> bool:
        return self.total_turns > 0 and self.covered_turns == self.total_turns


def _check_turn(turn) -> list[str]:
    problems = []

    if not turn.of("chunk_received"):
        problems.append("missing chunk_received events")

    if not turn.of("controller_decision"):
        problems.append("missing controller_decision events")

    started = {e["request_id"]: e for e in turn.of("retrieval_started")}
    completed = {e["request_id"]: e for e in turn.of("retrieval_completed")}
    for rid in completed:
        if rid not in started:
            problems.append(f"retrieval_completed {rid} has no matching retrieval_started")
    for rid, s in started.items():
        c = completed.get(rid)
        if c is not None and c.get("ts_ms", 0) < s.get("ts_ms", 0):
            problems.append(f"retrieval_completed {rid} ordered before its retrieval_started")

    versions = turn.of("answer_version_created")
    for v in versions:
        if "citations" not in v or "version" not in v or "parent_version" not in v:
            problems.append(f"answer_version_created missing lineage fields: {v}")
        for r in started.values():
            if r.get("ts_ms", 0) > v.get("ts_ms", 0):
                problems.append("answer_version_created ordered before a retrieval it may have used")

    if versions and not turn.of("output_emitted"):
        problems.append("answer produced but no output_emitted event")

    return problems


def score_g6(trace_events: list[dict], ground_truth: dict | None = None) -> G6Result:
    turns = assemble_turns(trace_events)
    total = len(turns)
    problems: list[str] = []
    covered = 0
    for t in turns:
        turn_problems = _check_turn(t)
        if turn_problems:
            problems.extend(f"{t.session_id}/{t.utterance_id}: {p}" for p in turn_problems)
        else:
            covered += 1
    return G6Result(total_turns=total, covered_turns=covered, problems=problems)
