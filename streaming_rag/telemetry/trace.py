"""Per-turn trace assembler: groups a flat event list into one record per
(session_id, utterance_id) turn, so the coverage checker and dashboard can
reconstruct "what happened" purely from the trace.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnTrace:
    session_id: str
    utterance_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def of(self, event_name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == event_name]

    def first(self, event_name: str) -> dict[str, Any] | None:
        matches = self.of(event_name)
        return matches[0] if matches else None

    def last(self, event_name: str) -> dict[str, Any] | None:
        matches = self.of(event_name)
        return matches[-1] if matches else None


def load_events(path: str) -> list[dict[str, Any]]:
    import json
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def assemble_turns(events: list[dict[str, Any]]) -> list[TurnTrace]:
    """Group events by (session_id, utterance_id). Session-scoped events with
    no utterance_id (session_started/session_ended) are attached to every
    turn in that session so per-turn checks can see them."""
    by_key: dict[tuple[str, str], TurnTrace] = {}
    order: list[tuple[str, str]] = []
    session_scoped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for e in events:
        sid = e.get("session_id")
        uid = e.get("utterance_id")
        if uid is None:
            if sid is not None:
                session_scoped[sid].append(e)
            continue
        key = (sid, uid)
        if key not in by_key:
            by_key[key] = TurnTrace(session_id=sid, utterance_id=uid)
            order.append(key)
        by_key[key].events.append(e)

    turns = [by_key[k] for k in order]
    for t in turns:
        t.events = session_scoped.get(t.session_id, []) + t.events
        t.events.sort(key=lambda e: (e.get("ts_ms") is None, e.get("ts_ms"), e.get("wall_ms", 0)))
    return turns
