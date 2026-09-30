"""H3: ONNX Runtime transformer graph fusion (fp32, same weights) for the three
production models, written next to the model cache:

    .cache/streaming_rag/models/fused/{e5,ce,reader}.onnx

Reports node counts before/after and the max |difference| of outputs vs the
original model on the same inputs (both on CUDA, and fused-CUDA vs
original-CPU = the reference configuration).

    python gpu_env.py handoff/gpu_tools/fuse_models.py
"""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers import optimizer

from streaming_rag.retrieval import neural

OUT = neural._CACHE_DIR / "models" / "fused"
MODELS = {
    "e5": (neural._resolve(neural.E5_SMALL_V2, "model.onnx"), neural._resolve(neural.E5_SMALL_V2, "tokenizer.json")),
    "ce": (neural._resolve(neural.MS_MARCO_MINILM_L6, "onnx/model.onnx"),
           neural._resolve(neural.MS_MARCO_MINILM_L6, "tokenizer.json")),
    "reader": (str(neural.READER_DIR / "model.onnx"), str(neural.READER_DIR / "tokenizer.json")),
}
Q = "which river flows through the city where the treaty was signed in 1713"
P = ("The Treaty of Utrecht was signed in 1713 in the Dutch city of Utrecht, through which the Kromme Rijn "
     "and the Vecht flow; it ended the War of the Spanish Succession. ")


def sess(path, cuda):
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=prov)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for key, (src, tok_path) in MODELS.items():
        opt = optimizer.optimize_model(src, model_type="bert", num_heads=12, hidden_size=384,
                                       use_gpu=True, opt_level=0, only_onnxruntime=False)
        dst = OUT / f"{key}.onnx"
        opt.save_model_to_file(str(dst))
        n_before = len(onnx.load(src).graph.node)
        n_after = len(onnx.load(str(dst)).graph.node)
        tok = neural._tokenizer(tok_path, 512 if key != "reader" else 384)
        inputs = {"e5": [f"query: {Q}", f"passage: {P * 4}"], "ce": [(Q, P * k) for k in (1, 2, 4, 6)],
                  "reader": [(Q, P * k) for k in (1, 3, 5)]}[key]
        enc = tok.encode_batch(inputs)
        ref_cpu = sess(src, False)
        ref_gpu, fused_gpu = sess(src, True), sess(str(dst), True)
        f = neural._feeds(ref_cpu, enc)
        f_fused = {k: v for k, v in f.items() if k in {i.name for i in fused_gpu.get_inputs()}}
        a, b, c = ref_cpu.run(None, f), ref_gpu.run(None, f), fused_gpu.run(None, f_fused)
        report[key] = {"nodes_before": n_before, "nodes_after": n_after,
                       "fused_ops": {k: v for k, v in opt.get_fused_operator_statistics().items() if v},
                       "maxdiff_origGPU_vs_origCPU": max(float(np.abs(x - y).max()) for x, y in zip(b, a)),
                       "maxdiff_fusedGPU_vs_origCPU": max(float(np.abs(x - y).max()) for x, y in zip(c, a)),
                       "maxdiff_fusedGPU_vs_origGPU": max(float(np.abs(x - y).max()) for x, y in zip(c, b)),
                       "fused_bound": fused_gpu.get_providers()[0]}
    print(json.dumps(report, indent=1))
    Path("handoff/newmachine/fuse_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
