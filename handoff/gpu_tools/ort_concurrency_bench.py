"""Screen CUDA EP options under the pipeline's concurrency pattern (hypothesis
test, not a pipeline measurement).

Shapes mimic the engine: e5 query encode (1 x ~20 tok), cross-encoder rerank
(5 x ~200), cross-encoder claim selection (~12 sentences x ~60), reader
(3 x 384). WORKERS threads each loop over a random mix of these calls, with
sequence lengths jittered so new shapes keep appearing (as with real text).
Per-call latency per model, p50 / p99 / max; the first call of each shape is
NOT excluded.

    python gpu_env.py handoff/gpu_tools/ort_concurrency_bench.py VARIANT [WORKERS]
      VARIANT  default | nocopy | nocopy_same | nocopy_same_warm | cpu | fused
"""
import json
import random
import sys
import threading
import time

import numpy as np
import onnxruntime as ort

from streaming_rag.retrieval import neural

VARIANT = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N_PER_WORKER = 150


def make(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    o.inter_op_num_threads = 1
    if VARIANT == "cpu":
        return ort.InferenceSession(path, sess_options=o, providers=["CPUExecutionProvider"])
    opts = {}
    if VARIANT.startswith("nocopy"):
        opts["do_copy_in_default_stream"] = False
    if "same" in VARIANT:
        opts["arena_extend_strategy"] = "kSameAsRequested"
    s = ort.InferenceSession(path, sess_options=o, providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider"
    return s


def feeds(sess, b, n):
    f = {"input_ids": np.random.randint(1000, 20000, (b, n)).astype(np.int64),
         "attention_mask": np.ones((b, n), np.int64)}
    if any(i.name == "token_type_ids" for i in sess.get_inputs()):
        f["token_type_ids"] = np.zeros((b, n), np.int64)
    return f


def main():
    if VARIANT.startswith("fused"):   # H3: fuse_models.py output
        fdir = neural._CACHE_DIR / "models" / "fused"
        e5, ce, rd = (make(str(fdir / f"{k}.onnx")) for k in ("e5", "ce", "reader"))
    else:
        e5 = make(neural._resolve(neural.E5_SMALL_V2, "model.onnx"))
        ce = make(neural._resolve(neural.MS_MARCO_MINILM_L6, "onnx/model.onnx"))
        rd = make(str(neural.READER_DIR / "model.onnx"))
    calls = {"e5_query": (e5, 1, (12, 32)), "ce_rerank": (ce, 5, (150, 260)),
             "ce_claim": (ce, 12, (40, 90)), "reader": (rd, 3, (300, 384))}
    if VARIANT.endswith("warm"):   # pre-grow arenas / load kernels at the largest shapes
        for s, b, (lo, hi) in calls.values():
            for n in (lo, hi, 512 if s is not rd else 384):
                s.run(None, feeds(s, max(b, 12), n))
    for s, b, (lo, hi) in calls.values():   # one ordinary warm-up call each (model load)
        s.run(None, feeds(s, b, lo))
    lat = {k: [] for k in calls}
    lock = threading.Lock()

    def worker(seed):
        rng = random.Random(seed)
        for _ in range(N_PER_WORKER):
            k = rng.choice(list(calls))
            s, b, (lo, hi) = calls[k]
            f = feeds(s, b, rng.randint(lo, hi))
            t = time.perf_counter()
            s.run(None, f)
            ms = (time.perf_counter() - t) * 1000
            with lock:
                lat[k].append(ms)

    t0 = time.perf_counter()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    out = {"variant": VARIANT, "workers": WORKERS, "wall_s": round(wall, 2)}
    for k, v in lat.items():
        a = np.array(v)
        out[k] = {"n": len(v), "p50": round(float(np.percentile(a, 50)), 2),
                  "p99": round(float(np.percentile(a, 99)), 2), "max": round(float(a.max()), 2)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
