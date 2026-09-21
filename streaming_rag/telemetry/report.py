"""Text dashboard: python -m streaming_rag.telemetry.report trace.jsonl

Prints, per turn: timeline of key events, retrieval triggers, answer
version lineage and token cost. No jsonschema/network dependency.
"""
from __future__ import annotations

import argparse
import sys

from .trace import load_events, assemble_turns
from .cost import total_cost_usd


def format_turn(turn) -> str:
    lines = [f"--- session={turn.session_id} utterance={turn.utterance_id} ---"]
    for e in turn.events:
        ts = e.get("ts_ms")
        ts_s = f"{ts:>7.0f}ms" if isinstance(ts, (int, float)) else "     n/a"
        ev = e.get("event", "?")
        extra_keys = [k for k in e.keys() if k not in ("event", "ts_ms", "wall_ms", "session_id", "utterance_id", "component")]
        extras = ", ".join(f"{k}={e[k]}" for k in extra_keys[:4])
        lines.append(f"  [{ts_s}] {ev:<24} {extras}")
    versions = turn.of("answer_version_created")
    if versions:
        last = versions[-1]
        lines.append(f"  => final version {last.get('version')} (parent {last.get('parent_version')}), citations={last.get('citations')}")
    cost = total_cost_usd(turn.events)
    lines.append(f"  cost: ${cost:.6f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Text dashboard for a streaming_rag trace file")
    parser.add_argument("trace_path")
    args = parser.parse_args(argv)

    events = load_events(args.trace_path)
    turns = assemble_turns(events)
    if not turns:
        print("(no turns found in trace)")
        return 0

    for turn in turns:
        print(format_turn(turn))
        print()

    total = total_cost_usd(events)
    print(f"===== {len(turns)} turn(s), total cost ${total:.6f} =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
