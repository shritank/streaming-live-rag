"""Optional audio front end: streaming Whisper (faster-whisper / CTranslate2)
that turns a 16 kHz mono waveform into the transcript-event stream the Engine
already consumes (transcript_chunk ... utterance_end).

Not a runtime requirement of the text pipeline: faster-whisper is imported
only when this is used. Chosen configuration (docs/experiment_log.md E4-E5):
base.en, int8, on the GPU when present - it keeps up with 1 s re-decodes on
an MX250, and gave +26 points answer recall over the dataset-provided ASR
transcripts on HeySQuAD dev. Streaming defaults: 1 s re-decode step, first
decode at 2 s, max_new_tokens=64.

    python -m streaming_rag.asr.whisper MODEL_DIR AUDIO.wav [--device cuda]
"""
from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .local_agreement import LocalAgreement


def load_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        t = np.linspace(0, len(x) / sr, int(len(x) * 16000 / sr), endpoint=False)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


class StreamingWhisper:
    """Re-decodes the growing buffer every `step_s` seconds of audio and
    commits words by LocalAgreement-2. Decodes run serially: decode k starts
    when its audio exists and the previous decode has finished, so a model
    that is too slow falls behind the speaker exactly as it would live."""

    def __init__(self, model_dir: str, device: str = "auto", compute_type: str = "int8",
                 step_s: float = 1.0, beam_size: int = 5, min_first_s: float = 2.0,
                 max_new_tokens: int = 64):
        from ..gpu import GpuUnavailable, add_cuda_dlls, asr_device
        # auto: the GPU whenever CTranslate2 sees one (STREAMING_RAG_ASR_DEVICE)
        device = asr_device() if device == "auto" else device
        if device == "cuda":
            add_cuda_dlls()
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model_dir, device=device, compute_type=compute_type,
                                   cpu_threads=int(os.environ.get("STREAMING_RAG_ASR_THREADS", "6")))
        if self._model.model.device != device:
            raise GpuUnavailable(f"Whisper loaded on {self._model.model.device}, requested {device}")
        self.device = device
        self._step = step_s
        self._beam = beam_size
        # Whisper hallucinates/loops on ~1 s fragments (up to its 448-token
        # limit: 12-17 s decodes); no decode before 2 s of audio, and a spoken
        # question is far below 64 tokens, so the cap only bounds runaway
        # generation (docs/experiment_log.md E6: decode p95 3.8 s -> 1.2 s,
        # transcripts identical to offline 83/100 -> 91/100).
        self._min_first = min_first_s
        self._max_tokens = max_new_tokens

    def _decode(self, audio: np.ndarray) -> list[str]:
        segs, _ = self._model.transcribe(audio, language="en", beam_size=self._beam,
                                         condition_on_previous_text=False, vad_filter=False,
                                         max_new_tokens=self._max_tokens)
        return " ".join(s.text.strip() for s in segs).split()

    def events(self, audio: np.ndarray, session_id: str = "s1", utterance_id: str = "u1") -> list[dict]:
        """Engine envelope events for one utterance, timestamped in audio
        time (ms from the start of speech) at which each chunk would really
        be available; utterance_end lands with the final flush."""
        dur = len(audio) / 16000
        cuts = [k * self._step for k in range(1, int(dur / self._step) + 1)
                if self._min_first <= k * self._step < dur] + [dur]
        agreement, clock, out = LocalAgreement(), 0.0, []
        for t in cuts:
            start = time.perf_counter()
            hyp = self._decode(audio[:int(t * 16000)])
            clock = max(clock, t) + (time.perf_counter() - start)
            new = agreement.flush(hyp) if t == dur else agreement.update(hyp)
            if new:
                first = len(agreement.committed) == len(new)
                out.append({"timestamp_ms": int(clock * 1000), "event_type": "transcript_chunk",
                            "payload": {"session_id": session_id, "utterance_id": utterance_id,
                                        "text": ("" if first else " ") + " ".join(new)}})
            if t != dur and len(hyp) > len(agreement.committed):
                # the full current hypothesis, uncommitted words included: lets
                # the engine start retrieval for the likely rest of the question
                # while the final decode is still running (never transcript text)
                out.append({"timestamp_ms": int(clock * 1000), "event_type": "transcript_hypothesis",
                            "payload": {"session_id": session_id, "utterance_id": utterance_id,
                                        "text": " ".join(hyp)}})
        out.append({"timestamp_ms": int(clock * 1000) + 1, "event_type": "utterance_end",
                    "payload": {"session_id": session_id, "utterance_id": utterance_id}})
        return out


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("audio")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args(argv)
    events = StreamingWhisper(args.model_dir, args.device, args.compute_type).events(load_wav(args.audio))
    print(json.dumps(events, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
