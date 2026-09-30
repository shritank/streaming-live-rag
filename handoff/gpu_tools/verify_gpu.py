"""Controlled GPU verification (run via gpu_env.py inside .venv-gpu).

For each production RAG model (e5, cross-encoder, SQuAD2 reader): build a CPU
session and a CUDA session with no CPU fallback allowed for the CUDA one,
check which provider the session actually bound, compare outputs, time both,
and read nvidia-smi before/after to confirm the process allocated GPU memory.
Then load Whisper base.en through CTranslate2 on cuda and decode one clip.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from streaming_rag.retrieval import neural


def smi() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu,driver_version,compute_cap",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    name, mem, util, drv, cc = [s.strip() for s in q.split(",")]
    return {"name": name, "mem_used_mib": int(mem), "util_pct": int(util), "driver": drv, "compute_cap": cc}


def sess(path, provider):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    o.inter_op_num_threads = 1
    if provider == "cuda":
        return ort.InferenceSession(path, sess_options=o, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])


def ep_split(path, feeds):
    """Run once with ORT's profiler and report, per execution provider, how
    many kernel executions and how much kernel time it actually took."""
    o = ort.SessionOptions()
    o.enable_profiling = True
    o.profile_file_prefix = str(Path(os.environ.get("TEMP", ".")) / "ortprof")
    s = ort.InferenceSession(path, sess_options=o, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    for _ in range(3):
        s.run(None, feeds)
    prof = Path(s.end_profiling())
    ev = json.loads(prof.read_text())
    prof.unlink()
    split = {}
    for e in ev:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            ep = e.get("args", {}).get("provider", "?")
            d = split.setdefault(ep, {"kernels": 0, "us": 0})
            d["kernels"] += 1
            d["us"] += int(e.get("dur", 0))
    return split


def timed(fn, reps=50, warm=5):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"p50": round(ts[len(ts) // 2], 2), "p99": round(ts[int(0.99 * (len(ts) - 1))], 2), "max": round(ts[-1], 2)}


def main():
    report = {"onnxruntime": ort.__version__, "available_providers": ort.get_available_providers(),
              "device": ort.get_device(), "smi_before": smi(), "models": {}}
    q = "what is the capital city of the region where the treaty was signed"
    p = ("The treaty was signed in the city of Utrecht, which lies in the province of Utrecht in the "
         "central Netherlands, and its capital is also called Utrecht. ") * 3
    specs = {
        "e5": (neural._resolve(neural.E5_SMALL_V2, "model.onnx"), neural._resolve(neural.E5_SMALL_V2, "tokenizer.json"), [f"query: {q}"]),
        "cross_encoder": (neural._resolve(neural.MS_MARCO_MINILM_L6, "onnx/model.onnx"),
                          neural._resolve(neural.MS_MARCO_MINILM_L6, "tokenizer.json"), [(q, p)] * 5),
        "reader": (str(neural.READER_DIR / "model.onnx"), str(neural.READER_DIR / "tokenizer.json"), [(q, p)] * 3),
    }
    for key, (mpath, tpath, inputs) in specs.items():
        tok = neural._tokenizer(tpath, 384)
        enc = tok.encode_batch(inputs)
        t = time.perf_counter()
        cpu = sess(mpath, "cpu")
        cpu_init = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        gpu = sess(mpath, "cuda")
        gpu_init = (time.perf_counter() - t) * 1000
        feeds = neural._feeds(cpu, enc)
        oc = cpu.run(None, feeds)
        og = gpu.run(None, feeds)
        diff = max(float(np.max(np.abs(a - b))) for a, b in zip(oc, og))
        report["models"][key] = {
            "bound_providers_gpu_session": gpu.get_providers(), "batch": len(inputs), "seq": len(enc[0].ids),
            "kernel_split_by_ep": ep_split(mpath, feeds),
            "max_abs_diff_cpu_vs_gpu": diff, "init_ms_cpu": round(cpu_init, 1), "init_ms_gpu": round(gpu_init, 1),
            "cpu_ms": timed(lambda: cpu.run(None, feeds)), "gpu_ms": timed(lambda: gpu.run(None, feeds))}
    report["smi_after_ort"] = smi()

    from faster_whisper import WhisperModel
    iso = Path("handoff/asr_tools")
    clip = sorted((iso / "heysquad_audio").glob("*.wav"))
    t = time.perf_counter()
    wm = WhisperModel(str(iso / "whisper" / "base.en"), device="cuda", compute_type="int8")
    w = {"init_ms": round((time.perf_counter() - t) * 1000, 1), "device": "cuda", "compute_type": "int8"}
    if clip:
        from streaming_rag.asr.whisper import load_wav
        x = load_wav(str(clip[0]))
        dur = len(x) / 16000
        segs, _ = wm.transcribe(x, language="en", beam_size=5, condition_on_previous_text=False, vad_filter=False)
        w["text"] = " ".join(s.text.strip() for s in segs)
        w["decode_ms"] = timed(lambda: list(wm.transcribe(x, language="en", beam_size=5,
                                                           condition_on_previous_text=False, vad_filter=False)[0]),
                               reps=10, warm=2)
        w["audio_s"] = round(dur, 2)
    import ctranslate2
    w["ctranslate2"] = ctranslate2.__version__
    w["ct2_cuda_device_count"] = ctranslate2.get_cuda_device_count()
    report["whisper"] = w
    report["smi_after_whisper"] = smi()
    print(json.dumps(report, indent=1))
    out = os.environ.get("VERIFY_OUT")
    if out:
        Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
