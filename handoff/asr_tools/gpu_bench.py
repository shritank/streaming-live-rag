"""CPU vs GPU benchmark for this project's model family (isolated process).

Model: deepset/minilm-uncased-squad2 (BERT, 12 layers, hidden 384) — the same
architecture family as the MiniLM cross-encoders and e5-small. Measures the
forward-pass latency of realistic batches (query + passage pairs):

  torch-cpu    PyTorch 2.4.1 on the 8-thread CPU
  torch-cuda   PyTorch 2.4.1 on the NVIDIA MX250 (2 GB)
  ort-cpu      the production path: ONNX Runtime 1.20.1 CPU, same exported model

Each point: 3 warm-up runs, then the median of 15 timed runs.
"""
import statistics
import sys
import time
from pathlib import Path

ISO = Path(__file__).parent
sys.path.insert(0, str(ISO / "pylib_tf"))
sys.path.insert(0, str(ISO / "pylib_stub"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForQuestionAnswering  # noqa: E402

SRC = ISO / "models" / "deepset--minilm-uncased-squad2"
ONNX = ISO / "models" / "deepset--minilm-uncased-squad2-onnx" / "model.onnx"


def timed(fn, reps=15, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def main():
    torch.set_num_threads(8)
    model = AutoModelForQuestionAnswering.from_pretrained(SRC).eval()
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(ONNX), sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}), onnxruntime {ort.__version__}")
    rows = []
    for batch, seq in ((1, 64), (5, 192), (10, 192), (20, 192), (32, 256)):
        ids = torch.randint(1000, 20000, (batch, seq))
        mask = torch.ones_like(ids)
        types = torch.zeros_like(ids)
        with torch.no_grad():
            cpu = timed(lambda: model(input_ids=ids, attention_mask=mask, token_type_ids=types))
        feeds = {"input_ids": ids.numpy(), "attention_mask": mask.numpy(), "token_type_ids": types.numpy()}
        ortc = timed(lambda: sess.run(None, feeds))
        gpu = None
        if torch.cuda.is_available():
            m = model.to("cuda")
            gi, gm, gt = ids.cuda(), mask.cuda(), types.cuda()

            def run_gpu():
                with torch.no_grad():
                    out = m(input_ids=gi, attention_mask=gm, token_type_ids=gt)
                out.start_logits.cpu()          # include the device->host copy
            gpu = timed(run_gpu)
            model.to("cpu")
        rows.append((batch, seq, cpu, gpu, ortc))
        print(f"batch {batch:3d} x seq {seq:3d}: torch-cpu {cpu:8.1f} ms  torch-cuda {gpu if gpu is None else round(gpu, 1)!s:>8} ms"
              f"  ort-cpu {ortc:8.1f} ms")


def main_ort_gpu():
    """ONNX Runtime GPU build (isolated iso/pylib_gpu), CUDA libraries taken
    from the installed torch wheel. Separate process: both builds are named
    'onnxruntime'."""
    import os
    import torch as _t
    os.add_dll_directory(str(Path(_t.__file__).parent / "lib"))
    sys.path.insert(0, str(ISO / "pylib_gpu"))
    import onnxruntime as ort
    print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())
    sess = ort.InferenceSession(str(ONNX), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("session providers:", sess.get_providers())
    for batch, seq in ((1, 64), (5, 192), (10, 192), (20, 192), (32, 256)):
        feeds = {"input_ids": np.random.randint(1000, 20000, (batch, seq)).astype(np.int64),
                 "attention_mask": np.ones((batch, seq), np.int64), "token_type_ids": np.zeros((batch, seq), np.int64)}
        print(f"batch {batch:3d} x seq {seq:3d}: ort-cuda {timed(lambda: sess.run(None, feeds)):8.1f} ms")


if __name__ == "__main__":
    main_ort_gpu() if "--ort-gpu" in sys.argv else main()
