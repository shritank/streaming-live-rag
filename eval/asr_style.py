"""python -m eval.asr_style --src DIR --out DIR

Robustness probe: rewrites every transcript chunk the way many streaming ASR
systems emit text — lowercase, no punctuation — leaving timing, chunking and
ground truth untouched. The controller leans on capitalisation (entity
anchors), sentence punctuation (closure) and commas (clause splitting), so
this measures how much of its behaviour survives without those cues.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_PUNCT_RE = re.compile(r"[^\w\s']", re.UNICODE)


def asr_style(text: str) -> str:
    stripped = _PUNCT_RE.sub(" ", text.lower())
    lead = " " if text[:1].isspace() else ""
    return lead + " ".join(stripped.split())


def convert(src: Path, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.glob("*.json")):
        scenario = json.loads(path.read_text(encoding="utf-8"))
        for event in scenario["events"]:
            if event.get("event_type") == "transcript_chunk":
                event["payload"]["text"] = asr_style(event["payload"].get("text", ""))
        scenario["scenario_id"] = scenario.get("scenario_id", path.stem) + "_asr"
        (out / path.name).write_text(json.dumps(scenario, indent=2), encoding="utf-8")
        n += 1
    return n


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    print(f"wrote {convert(Path(args.src), Path(args.out))} scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
