"""Transcribe the downloaded HeySQuAD clips with one faster-whisper model.

    python transcribe.py MODEL DEVICE COMPUTE OUT.json [--limit N]
      MODEL    tiny.en | base.en | small.en   (iso/whisper/<MODEL>, pinned)
      DEVICE   cpu | cuda
      COMPUTE  int8 | float32 | int8_float32 ...

Writes {id: transcript} plus per-clip latency, audio seconds, and peak RSS.
Decoding: beam 5, English, no previous-text conditioning (each clip is an
independent question), no VAD (clips are already single utterances).
"""
import json
import os
import sys
import time
import wave
from pathlib import Path

ISO = Path(__file__).parent
sys.path[:0] = [str(ISO / "pylib_fw"), str(ISO / "pylib_stub")]
if sys.argv[2] == "cuda":
    try:  # old laptop: CUDA 12 / cuDNN 9 DLLs from a torch wheel; new workstation: gpu_tools/gpu_env.py
        import torch  # noqa: F401
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    except ImportError:
        pass

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


def load_wav(path: Path) -> tuple[np.ndarray, float]:
    with wave.open(str(path)) as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    assert width == 2, f"{path}: expected 16-bit PCM"
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:  # linear resample; clips are speech, fine for ASR
        t = np.linspace(0, len(x) / sr, int(len(x) * 16000 / sr), endpoint=False)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return x, len(x) / 16000


def peak_rss_mb() -> float:
    """Peak working set of this process (Windows GetProcessMemoryInfo)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


def main() -> int:
    model_name, device, compute, out = sys.argv[1:5]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    clips = sorted((ISO / "heysquad_audio").glob("*.wav"))[:limit]
    t0 = time.perf_counter()
    model = WhisperModel(str(ISO / "whisper" / model_name), device=device, compute_type=compute,
                         cpu_threads=int(os.environ.get("ASR_THREADS", "6")))
    load_s = time.perf_counter() - t0
    res, audio_s, dec_s = {}, 0.0, 0.0
    for i, p in enumerate(clips):
        x, dur = load_wav(p)
        t = time.perf_counter()
        segs, _ = model.transcribe(x, language="en", beam_size=5, condition_on_previous_text=False,
                                   vad_filter=False)
        text = " ".join(s.text.strip() for s in segs).strip()
        dt = time.perf_counter() - t
        res[p.stem] = {"text": text, "sec": round(dt, 3), "audio_sec": round(dur, 2)}
        audio_s += dur
        dec_s += dt
        if i % 100 == 0:
            print(f"{i}/{len(clips)} rtf={dec_s / max(audio_s, 1e-9):.3f}", flush=True)
    meta = {"model": model_name, "revision": (ISO / "whisper" / model_name / "REVISION").read_text().strip(),
            "device": device, "compute_type": compute, "n": len(res), "load_s": round(load_s, 1),
            "decode_s": round(dec_s, 1), "audio_s": round(audio_s, 1), "rtf": round(dec_s / max(audio_s, 1e-9), 3),
            "mean_ms_per_clip": round(1000 * dec_s / max(len(res), 1), 1), "peak_rss_mb": round(peak_rss_mb(), 0)}
    Path(out).write_text(json.dumps({"meta": meta, "items": res}, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
