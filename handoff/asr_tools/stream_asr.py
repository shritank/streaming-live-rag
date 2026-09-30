"""Streaming Whisper by LocalAgreement-2 (as in ufal/whisper_streaming):
re-decode the growing audio prefix every STEP seconds and commit only the
words two consecutive hypotheses agree on, so the output is append-only (our
controller cannot retract words). When the audio ends, the final decode is
flushed.

Decodes run serially in audio time: decode k starts at max(t_k, end of decode
k-1), so a slow model falls behind the speaker exactly as it would live.

    python stream_asr.py MODEL DEVICE COMPUTE OUT.json [--step 1.0] [--limit N]

Per clip: [(ms, committed_text_delta)], final text, audio duration, and
asr_post_speech_ms = time the final flush lands minus the end of the audio.
"""
import json
import os
import sys
import time
from pathlib import Path

ISO = Path(__file__).parent
sys.path[:0] = [str(ISO / "pylib_fw"), str(ISO / "pylib_stub")]
if sys.argv[2] == "cuda":
    try:  # old laptop: DLLs from a torch wheel; new workstation: gpu_tools/gpu_env.py
        import torch  # noqa: F401
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    except ImportError:
        pass

from faster_whisper import WhisperModel  # noqa: E402

from transcribe import load_wav  # noqa: E402


def lcp(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    model_name, device, compute, out = sys.argv[1:5]
    step = float(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 1.0
    min_first = float(sys.argv[sys.argv.index("--min-first") + 1]) if "--min-first" in sys.argv else step
    max_tok = int(sys.argv[sys.argv.index("--max-tokens") + 1]) if "--max-tokens" in sys.argv else None
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    clips = sorted((ISO / "heysquad_audio").glob("*.wav"))
    if "--ids" in sys.argv:
        keep = set(Path(sys.argv[sys.argv.index("--ids") + 1]).read_text().split())
        clips = [c for c in clips if c.stem in keep]
    clips = clips[:limit]
    model = WhisperModel(str(ISO / "whisper" / model_name), device=device, compute_type=compute,
                         cpu_threads=int(os.environ.get("ASR_THREADS", "6")))

    def decode(x):
        t = time.perf_counter()
        kw = {"max_new_tokens": max_tok} if max_tok else {}
        segs, _ = model.transcribe(x, language="en", beam_size=5, condition_on_previous_text=False,
                                   vad_filter=False, **kw)
        text = " ".join(s.text.strip() for s in segs).strip()
        return text.split(), time.perf_counter() - t

    decode(load_wav(clips[0])[0][:16000])   # warm-up, excluded from timing
    res, post = {}, []
    for i, p in enumerate(clips):
        x, dur = load_wav(p)
        cuts = [k * step for k in range(1, int(dur / step) + 1) if min_first <= k * step < dur] + [dur]
        committed, prev, events, clock, decode_ms, hyps = [], None, [], 0.0, [], []
        for t in cuts:
            hyp, dt = decode(x[:int(t * 16000)])
            decode_ms.append(int(dt * 1000))
            clock = max(clock, t) + dt                     # serial decoder, cannot start before the audio exists
            hyps.append([int(clock * 1000), " ".join(hyp)])  # (ms available, full hypothesis)
            if t == dur:                                   # audio over: flush the final hypothesis
                new = hyp[len(committed):] if hyp[:len(committed)] == committed else hyp[lcp(committed, hyp):]
            else:
                agreed = lcp(prev, hyp) if prev is not None else 0
                new = hyp[len(committed):agreed] if agreed > len(committed) else []
            if new:
                events.append([int(clock * 1000), (" " if committed else "") + " ".join(new)])
                committed += new
            prev = hyp
        post_ms = int((clock - dur) * 1000)
        post.append(post_ms)
        res[p.stem] = {"events": events, "text": " ".join(committed), "audio_ms": int(dur * 1000),
                       "asr_post_speech_ms": post_ms, "n_decodes": len(cuts),
                       "decode_ms": decode_ms, "hypotheses": hyps}
        if i % 100 == 0:
            print(f"{i}/{len(clips)}", flush=True)
    post.sort()
    meta = {"model": model_name, "device": device, "compute_type": compute, "step_s": step,
            "min_first_s": min_first, "max_new_tokens": max_tok, "n": len(res),
            "asr_post_speech_ms_p50": post[len(post) // 2], "asr_post_speech_ms_p95": post[int(0.95 * (len(post) - 1))]}
    Path(out).write_text(json.dumps({"meta": meta, "items": res}, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
