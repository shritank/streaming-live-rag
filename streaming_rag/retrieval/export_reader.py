"""One-time build step: convert the SQuAD2 reader to ONNX.

    python -m streaming_rag.retrieval.export_reader

Downloads deepset/minilm-uncased-squad2 at a pinned revision, checks the
weights' SHA-256, exports the question-answering head to ONNX and refuses to
install the result unless ONNX and PyTorch outputs agree to 1e-3. The
runtime never imports this module: only the exported model.onnx and
tokenizer.json (in neural.READER_DIR) are needed to run.

Build-time requirements (not in requirements.lock; the Dockerfile installs
them in a separate build stage): torch 2.4, transformers 4.46.3,
huggingface_hub 0.26.x, tokenizers 0.20.x, onnx 1.17.
"""
from __future__ import annotations

import hashlib
import sys

REPO = "deepset/minilm-uncased-squad2"
REVISION = "934656cdda79824eabf503ed56e15c01ddbdbe3f"
WEIGHTS_SHA256 = "240478c7251d5e55be35feb47177054b960259b9328af662ea3cee87ce9755a6"
FILES = ("config.json", "model.safetensors", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json")


def main() -> int:
    import numpy as np
    import onnxruntime as ort
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    from .neural import READER_DIR

    paths = {f: hf_hub_download(REPO, f, revision=REVISION) for f in FILES}
    digest = hashlib.sha256(open(paths["model.safetensors"], "rb").read()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise SystemExit(f"weights sha256 {digest} != pinned {WEIGHTS_SHA256}")
    src = str(__import__("pathlib").Path(paths["config.json"]).parent)
    tok = AutoTokenizer.from_pretrained(src, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(src).eval()
    READER_DIR.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(READER_DIR)

    enc = tok("What did Microsoft announce that it would rename SkyDrive to?",
              "On 27 January 2014, Microsoft announced that SkyDrive would be renamed OneDrive.",
              return_tensors="pt")
    with torch.no_grad():
        ref = model(**enc)
    target = READER_DIR / "model.onnx"
    torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]), str(target),
                      input_names=["input_ids", "attention_mask", "token_type_ids"],
                      output_names=["start_logits", "end_logits"],
                      dynamic_axes={n: {0: "batch", 1: "seq"} for n in
                                    ("input_ids", "attention_mask", "token_type_ids", "start_logits", "end_logits")},
                      opset_version=17)
    sess = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: enc[k].numpy() for k in ("input_ids", "attention_mask", "token_type_ids")})
    diff = max(float(np.abs(got[0] - ref.start_logits.numpy()).max()),
               float(np.abs(got[1] - ref.end_logits.numpy()).max()))
    if diff >= 1e-3:
        target.unlink()
        raise SystemExit(f"ONNX export does not match PyTorch (max |diff| = {diff:.2e})")
    print(f"exported {REPO}@{REVISION[:8]} -> {target} (max |onnx - torch| = {diff:.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
