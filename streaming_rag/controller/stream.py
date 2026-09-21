"""Incremental transcript chunking simulator.

Turns a whole utterance into timestamped fragments at speaking speed, so the
CLI's live mode and the scenario generator both produce the same event shape
the real ASR stream would.
"""
from __future__ import annotations

from ..retrieval.text import tokenize

WORDS_PER_MINUTE = 150


def chunk_utterance(text: str, utterance_id: str, session_id: str, start_ms: int = 0,
                    words_per_chunk: int = 6, wpm: int = WORDS_PER_MINUTE) -> list[dict]:
    """Returns envelope events: transcript_chunk* followed by utterance_end."""
    words = text.split()
    if not words:
        return [{"timestamp_ms": start_ms, "event_type": "utterance_end",
                  "payload": {"session_id": session_id, "utterance_id": utterance_id}}]

    ms_per_word = 60_000 / wpm
    events = []
    ts = start_ms
    for i in range(0, len(words), words_per_chunk):
        group = words[i:i + words_per_chunk]
        fragment = (" " if i else "") + " ".join(group)
        events.append({
            "timestamp_ms": int(ts),
            "event_type": "transcript_chunk",
            "payload": {"session_id": session_id, "utterance_id": utterance_id, "text": fragment},
        })
        ts += ms_per_word * len(group)

    events.append({
        "timestamp_ms": int(ts + 300),
        "event_type": "utterance_end",
        "payload": {"session_id": session_id, "utterance_id": utterance_id},
    })
    return events
