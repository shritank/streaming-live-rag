"""Answer a spoken question from a WAV file: audio -> streaming Whisper -> engine -> cited answer.

    python tools/answer_audio.py CLIP.wav [CLIP2.wav ...]
        [--corpus-dir data/corpus] [--model handoff/asr_tools/whisper/small.en]
        [--device auto|cuda|cpu] [--time-scale 1] [--no-warmup] [--raw]

Prints the guide's structured output event record (retrieval_events, sub_queries,
answer, citations, uncertainty) on stdout, one per turn; the transcript Whisper
heard goes to stderr. --raw prints the engine's full turn_result instead.

The audio front end is outside the official scope (transcript stream -> RAG); this
is a convenience wrapper around streaming_rag.asr.whisper and the normal replay
path. The clip must be a WAV file (16-bit PCM; other sample rates and stereo are
converted). The corpus must be the one the question is about.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streaming_rag.asr.whisper import StreamingWhisper, load_wav  # noqa: E402
from streaming_rag.cli import print_guide_records, run_replay  # noqa: E402


def transcript_of(events: list[dict]) -> str:
    return "".join(e["payload"]["text"] for e in events if e["event_type"] == "transcript_chunk").strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--model", default="handoff/asr_tools/whisper/small.en")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--time-scale", type=float, default=1.0, help="1 = real time; higher replays faster")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--raw", action="store_true", help="print the full turn_result, not the guide record")
    args = ap.parse_args(argv)

    missing = [w for w in args.wav if not Path(w).is_file()]
    if missing:
        print(f"file not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    whisper = StreamingWhisper(args.model, device=args.device)
    if not args.no_warmup:
        whisper.events(load_wav(args.wav[0]))          # uncounted: first-call GPU start-up
    print(f"# Whisper {Path(args.model).name} on {whisper.device}, corpus {args.corpus_dir}", file=sys.stderr)

    for wav in args.wav:
        events = whisper.events(load_wav(wav))
        scenario = {"scenario_id": Path(wav).stem,
                    "events": [{"timestamp_ms": 0, "event_type": "session_start",
                                "payload": {"session_id": "s1"}}] + events}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.json"
            path.write_text(json.dumps(scenario), encoding="utf-8")
            results = asyncio.run(run_replay(str(path), args.time_scale, "streaming", None,
                                             corpus_dir=args.corpus_dir))
        print(f"# {Path(wav).name} - heard: {transcript_of(events)}", file=sys.stderr)
        if args.raw:
            for r in results:
                print(json.dumps(r, default=str))
        else:
            print_guide_records(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
