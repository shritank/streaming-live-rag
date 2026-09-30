"""Controlled GPU check through the PRODUCT code path (run in .venv-gpu,
defaults only - no provider env vars set, so `auto` must pick CUDA itself):

  * neural.get_e5_encoder / get_cross_encoder / get_reader: which graph file,
    which execution provider each session bound, one real call each;
  * per-operator placement of every model (ORT profiler): op types that ran
    on the CPU EP and their share of kernel time;
  * asr.whisper.StreamingWhisper(device="auto"): device CTranslate2 loaded on,
    one real streaming decode.

    .venv-gpu\\Scripts\\python handoff\\gpu_tools\\verify_product_gpu.py
"""
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from streaming_rag.asr.whisper import StreamingWhisper, load_wav  # noqa: E402
from streaming_rag.retrieval import neural  # noqa: E402


def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


def op_placement(path, feeds):
    o = ort.SessionOptions()
    o.enable_profiling = True
    o.profile_file_prefix = str(Path(os.environ.get("TEMP", ".")) / "ortprof")
    s = ort.InferenceSession(path, sess_options=o, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    for _ in range(3):
        s.run(None, feeds)
    prof = Path(s.end_profiling())
    ev = json.loads(prof.read_text())
    prof.unlink()
    us, cpu_ops = Counter(), Counter()
    for e in ev:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            ep = e["args"].get("provider")
            us[ep] += int(e.get("dur", 0))
            if ep == "CPUExecutionProvider":
                cpu_ops[e["args"].get("op_name")] += int(e.get("dur", 0))
    total = sum(us.values()) or 1
    return {"kernel_time_share": {k: round(v / total, 3) for k, v in us.items()},
            "cpu_op_types_us": dict(cpu_ops.most_common())}


def main():
    for v in ("STREAMING_RAG_ORT_PROVIDER", "STREAMING_RAG_ORT_FUSED", "STREAMING_RAG_ASR_DEVICE"):
        assert v not in os.environ, f"{v} set - this check must exercise the defaults"
    report = {"gpu_before": smi(), "onnxruntime": ort.__version__}
    q = "which river flows through the city where the treaty was signed"
    p = "The Treaty of Utrecht was signed in 1713 in Utrecht, through which the Vecht flows."
    e5, ce, rd = neural.get_e5_encoder(), neural.get_cross_encoder(), neural.get_reader()
    report["calls"] = {"e5_norm": float(np.linalg.norm(e5.encode([q])[0])),
                       "ce_logit": float(ce.score(q, [p])[0]),
                       "reader": rd.answer(q, [p])}
    report["sessions"] = dict(neural.ACTIVE_PROVIDERS)
    placement = {}
    for path in neural.ACTIVE_PROVIDERS:
        s = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        names = {i.name for i in s.get_inputs()}
        f = {"input_ids": np.full((5, 192), 1000, np.int64), "attention_mask": np.ones((5, 192), np.int64)}
        if "token_type_ids" in names:
            f["token_type_ids"] = np.zeros((5, 192), np.int64)
        placement[Path(path).name if "fused" not in path else "fused/" + Path(path).name] = op_placement(path, f)
    report["op_placement"] = placement
    whisper_dir = Path("handoff/asr_tools/whisper/base.en")
    clip = sorted(Path("handoff/asr_tools/heysquad_audio").glob("*.wav"))[0]
    w = StreamingWhisper(str(whisper_dir))
    ev = w.events(load_wav(str(clip)))
    report["whisper"] = {"device": w.device, "loaded_on": w._model.model.device,
                         "text": "".join(e["payload"].get("text", "") for e in ev)}
    report["gpu_after"] = smi()
    print(json.dumps(report, indent=1))
    Path("handoff/newmachine/gpu_verify_product.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
