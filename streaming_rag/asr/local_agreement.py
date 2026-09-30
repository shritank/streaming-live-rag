"""LocalAgreement-2 commit policy for a re-decoding (non-streaming) ASR.

Whisper-style models transcribe a finished buffer and may revise earlier
words as more audio arrives. The controller is append-only (it accumulates
chunk text and cannot retract words), so words are committed only once two
consecutive decodes of the growing buffer agree on them; when the audio ends
the final hypothesis is flushed. Same policy as ufal/whisper_streaming.

Pure logic, no model dependency: feed it one hypothesis per decode.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class LocalAgreement:
    committed: list[str] = field(default_factory=list)
    _previous: list[str] | None = None

    def update(self, hypothesis: list[str]) -> list[str]:
        """Words newly committed by this decode of a still-growing buffer."""
        agreed = _common_prefix(self._previous, hypothesis) if self._previous is not None else 0
        self._previous = hypothesis
        new = hypothesis[len(self.committed):agreed] if agreed > len(self.committed) else []
        self.committed += new
        return new

    def flush(self, final_hypothesis: list[str]) -> list[str]:
        """The audio has ended: commit the rest of the final decode.

        Committed words cannot be retracted. If the final decode revises
        some of them, its whole revised tail (from the first disagreement)
        is appended after them, so the corrected words still reach retrieval
        alongside the stale ones rather than being lost."""
        keep = _common_prefix(self.committed, final_hypothesis)
        new = final_hypothesis[len(self.committed):] if keep == len(self.committed) else final_hypothesis[keep:]
        self.committed += new
        return new
