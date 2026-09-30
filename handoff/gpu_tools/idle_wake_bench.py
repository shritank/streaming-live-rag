"""H4: does a CUDA call after the GPU has idled cost more than a steady-state
call? (Speech has pauses; the engine's GPU work comes in bursts.)

For each idle gap: sleep(gap), then time 3 back-to-back cross-encoder rerank
calls (5 x 200 tokens); repeated 12 times. Also samples the SM clock
reported by nvidia-smi right before the first call.

    python gpu_env.py handoff/gpu_tools/idle_wake_bench.py
"""
import json
import subprocess
import time

import numpy as np
import onnxruntime as ort

from streaming_rag.retrieval import neural


def sm_clock():
    q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,pstate", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    return q


def main():
    s = ort.InferenceSession(neural._resolve(neural.MS_MARCO_MINILM_L6, "onnx/model.onnx"),
                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider"
    f = {"input_ids": np.random.randint(1000, 20000, (5, 200)).astype(np.int64),
         "attention_mask": np.ones((5, 200), np.int64), "token_type_ids": np.zeros((5, 200), np.int64)}
    for _ in range(20):
        s.run(None, f)
    out = {}
    for gap in (0.0, 0.5, 2.0, 5.0):
        first, rest, clocks = [], [], []
        for _ in range(12):
            time.sleep(gap)
            if gap:
                clocks.append(sm_clock())
                time.sleep(0.05)
            ts = []
            for _ in range(3):
                t = time.perf_counter()
                s.run(None, f)
                ts.append((time.perf_counter() - t) * 1000)
            first.append(ts[0])
            rest += ts[1:]
        out[f"gap_{gap}s"] = {"first_call_p50": round(float(np.median(first)), 2),
                              "first_call_max": round(max(first), 2),
                              "next_calls_p50": round(float(np.median(rest)), 2),
                              "clock_before_first(sample)": clocks[:3]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
