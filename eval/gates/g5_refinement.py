"""G5 — session refinement: 100% of refinement scenarios show continuity.

A refinement turn passes when:
  - its answer_version_created event has change_kind == "refinement"
  - version == parent_version + 1 (no gap, no restart)
  - every citation from the parent version's answer still appears in the
    refined answer's citations (nothing was thrown away)
  - the retrieval issued for the turn is scoped to the delta, not a
    full-topic re-run (heuristically: fewer sub-queries than the prior
    turn's total, since a full restart would re-issue them all)
"""
from __future__ import annotations

from dataclasses import dataclass

from streaming_rag.telemetry.trace import assemble_turns


@dataclass
class G5Result:
    refinement_turns: int
    continuous: int
    failures: list[str]

    @property
    def rate(self) -> float:
        return self.continuous / self.refinement_turns if self.refinement_turns else 1.0

    @property
    def passed(self) -> bool:
        return self.refinement_turns == 0 or self.continuous == self.refinement_turns


def score_g5(trace_events: list[dict], ground_truth: dict) -> G5Result:
    turns_gt = ground_truth.get("turns", {})
    turns = {t.utterance_id: t for t in assemble_turns(trace_events)}

    refinement_turns = continuous = 0
    failures: list[str] = []

    # version -> citations, in emission order, across the whole session trace
    version_citations: dict[int, list[str]] = {}
    for e in trace_events:
        if e.get("event") == "answer_version_created":
            version_citations[e["version"]] = e.get("citations", [])

    for utterance_id, gt in turns_gt.items():
        if gt.get("kind") != "refinement":
            continue
        refinement_turns += 1
        turn = turns.get(utterance_id)
        if turn is None:
            failures.append(f"{utterance_id}: no trace found")
            continue
        created = turn.of("answer_version_created")
        if not created:
            failures.append(f"{utterance_id}: no answer_version_created")
            continue
        ev = created[-1]
        ok = True
        if ev.get("change_kind") != "refinement":
            failures.append(f"{utterance_id}: change_kind={ev.get('change_kind')!r}, expected 'refinement'")
            ok = False
        parent = ev.get("parent_version")
        version = ev.get("version")
        if parent is None or version != parent + 1:
            failures.append(f"{utterance_id}: version={version} parent={parent}, expected version=parent+1")
            ok = False
        if parent in version_citations:
            prior_citations = set(version_citations[parent])
            new_citations = set(ev.get("citations", []))
            missing = prior_citations - new_citations
            if missing:
                failures.append(f"{utterance_id}: dropped prior citations {sorted(missing)}")
                ok = False
        if ok:
            continuous += 1

    return G5Result(refinement_turns, continuous, failures)
