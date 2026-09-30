"""One-time ONNX export of deepset/minilm-uncased-squad2 (isolated toolchain).

Runs with the isolated transformers 4.46.3 / huggingface_hub 0.26 /
tokenizers 0.20 in iso/pylib_tf placed FIRST on sys.path, so the production
packages are never modified or shadowed outside this process. Verifies the
ONNX graph against the PyTorch model before declaring success.
"""
import sys
from pathlib import Path

ISO = Path(__file__).parent
sys.path.insert(0, str(ISO / "pylib_tf"))
sys.path.insert(0, str(ISO / "pylib_stub"))   # _lzma stand-in, this process only

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForQuestionAnswering, AutoTokenizer  # noqa: E402

SRC = ISO / "models" / "deepset--minilm-uncased-squad2"
OUT = ISO / "models" / "deepset--minilm-uncased-squad2-onnx"
OUT.mkdir(parents=True, exist_ok=True)

print("transformers", transformers.__version__, "torch", torch.__version__)
tok = AutoTokenizer.from_pretrained(SRC, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(SRC).eval()
tok.save_pretrained(OUT)

q = "What did Microsoft announce that it would rename SkyDrive to?"
c = ("In July 2013, the English High Court of Justice found that Microsoft's use of the term SkyDrive "
     "infringed on Sky's right to the Sky trademark. On 27 January 2014, Microsoft announced that "
     "SkyDrive would be renamed OneDrive.")
enc = tok(q, c, return_tensors="pt", truncation=True, max_length=384)
with torch.no_grad():
    ref = model(**enc)
torch.onnx.export(
    model, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]), str(OUT / "model.onnx"),
    input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["start_logits", "end_logits"],
    dynamic_axes={n: {0: "batch", 1: "seq"} for n in ("input_ids", "attention_mask", "token_type_ids",
                                                     "start_logits", "end_logits")},
    opset_version=17)

import onnxruntime as ort  # noqa: E402  (whichever onnxruntime this process sees; CPU provider)
sess = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
got = sess.run(None, {k: enc[k].numpy() for k in ("input_ids", "attention_mask", "token_type_ids")})
d_start = float(np.abs(got[0] - ref.start_logits.numpy()).max())
d_end = float(np.abs(got[1] - ref.end_logits.numpy()).max())
s, e = int(ref.start_logits.argmax()), int(ref.end_logits.argmax())
print("torch answer:", tok.decode(enc["input_ids"][0][s:e + 1]))
print(f"max |onnx - torch|: start {d_start:.2e}, end {d_end:.2e}")
assert d_start < 1e-3 and d_end < 1e-3, "ONNX export does not match PyTorch"
print("EXPORT OK ->", OUT)
