"""Neural retrieval components, run with ONNX Runtime + `tokenizers` only.

No torch/transformers at runtime: the official ONNX exports published in each
model's own Hugging Face repo are loaded directly. That keeps the dependency
footprint small and CPU inference fast enough for the streaming path.

  E5Embedder      intfloat/e5-small-v2 bi-encoder (384-d, mean pooling,
                  "query: " / "passage: " prefixes as the model card requires)
  CrossEncoder    cross-encoder/ms-marco-MiniLM-L6-v2 (query, passage) -> logit

Weights are pinned to exact repo revisions. Fetch them once with
    python -m streaming_rag.retrieval.neural --fetch
after which everything runs offline (files are read with local_files_only).
"""
from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    onnx_file: str
    sha256: tuple[tuple[str, str], ...]   # (file, SHA-256 of its content)

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(f for f, _ in self.sha256)


E5_SMALL_V2 = ModelSpec(
    repo="intfloat/e5-small-v2",
    revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
    onnx_file="model.onnx",
    sha256=(("config.json", "5dfb0363cd0243be179c03bcaafd1542d0fbb95e8cbcf575fff3e229342adc2f"),
            ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
            ("1_Pooling/config.json", "987f7a67a38fa564c849bb5d277c52ab9088a84368fc0be31a354125aebb12a0"),
            ("model.onnx", "4b8205be2a3c5fc53c6534d76a2012064f7309c162b806f2889c6ec8ec4fdcba")),
)
MS_MARCO_MINILM_L6 = ModelSpec(
    repo="cross-encoder/ms-marco-MiniLM-L6-v2",
    revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
    onnx_file="onnx/model.onnx",
    sha256=(("config.json", "380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc"),
            ("tokenizer.json", "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
            ("onnx/model.onnx", "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a")),
)

_CACHE_DIR = Path(os.environ.get("STREAMING_RAG_CACHE", ".cache/streaming_rag"))
_lock = threading.Lock()
_loaded: dict[str, object] = {}


class ModelNotAvailable(RuntimeError):
    pass


def _resolve(spec: ModelSpec, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(spec.repo, filename, revision=spec.revision, local_files_only=True)
    except Exception as e:  # missing from the local cache
        raise ModelNotAvailable(
            f"{spec.repo}@{spec.revision[:8]} file {filename!r} is not in the local Hugging Face "
            f"cache. Run: python -m streaming_rag.retrieval.neural --fetch") from e


def _session(path: str):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.environ.get("STREAMING_RAG_ORT_THREADS", "4"))
    opts.inter_op_num_threads = 1
    # cpu (default, the verified configuration) | cuda (needs the separate
    # onnxruntime-gpu build and CUDA 12 libraries; falls back to CPU if absent)
    providers = ["CPUExecutionProvider"]
    if os.environ.get("STREAMING_RAG_ORT_PROVIDER", "cpu") == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


def _tokenizer(path: str, max_length: int):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    return tok


def _feeds(session, encodings) -> dict:
    ids = np.array([e.ids for e in encodings], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    if any(i.name == "token_type_ids" for i in session.get_inputs()):
        feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
    return feeds


class E5Encoder:
    """The shared, stateless model (one per process)."""

    def __init__(self, spec: ModelSpec = E5_SMALL_V2, max_length: int = 512, batch_size: int = 32):
        self.spec = spec
        self.dim = 384
        self._session = _session(_resolve(spec, spec.onnx_file))
        self._tok = _tokenizer(_resolve(spec, "tokenizer.json"), max_length)
        self._batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self._batch_size):
            batch = self._tok.encode_batch(texts[start:start + self._batch_size])
            feeds = _feeds(self._session, batch)
            hidden = self._session.run(None, feeds)[0]                  # (b, seq, 384)
            mask = feeds["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            out.append(pooled)
        vecs = np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)


class E5Embedder:
    """Per-corpus object satisfying the retrieval `Embedder` protocol once
    `fit()` has encoded the corpus. Document vectors are cached on disk keyed
    by model revision and a hash of the exact indexed texts."""

    def __init__(self, encoder: E5Encoder | None = None):
        self._encoder = encoder or get_e5_encoder()
        self.dim = self._encoder.dim
        self._doc_vectors: np.ndarray | None = None

    def fit(self, documents: list[str]) -> "E5Embedder":
        digest = hashlib.sha256("\x1e".join(documents).encode("utf-8")).hexdigest()[:24]
        rev = self._encoder.spec.revision[:8]
        cache = _CACHE_DIR / "embeddings" / f"e5-small-v2@{rev}" / f"{digest}.npy"
        if cache.exists():
            vecs = np.load(cache)
            if vecs.shape == (len(documents), self.dim):
                self._doc_vectors = vecs
                return self
        self._doc_vectors = self._encoder.encode([f"passage: {d}" for d in documents])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, self._doc_vectors)
        return self

    @property
    def doc_vectors(self) -> np.ndarray:
        if self._doc_vectors is None:
            raise RuntimeError("E5Embedder.fit() must be called before use")
        return self._doc_vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._encoder.encode([f"query: {t}" for t in texts])


class CrossEncoder:
    """Scores (query, passage) pairs; higher logit = more relevant."""

    def __init__(self, spec: ModelSpec = MS_MARCO_MINILM_L6, max_length: int = 512, batch_size: int = 32):
        self.spec = spec
        self._session = _session(_resolve(spec, spec.onnx_file))
        self._tok = _tokenizer(_resolve(spec, "tokenizer.json"), max_length)
        self._batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        scores = []
        for start in range(0, len(passages), self._batch_size):
            batch = self._tok.encode_batch([(query, p) for p in passages[start:start + self._batch_size]])
            logits = self._session.run(None, _feeds(self._session, batch))[0]
            scores.append(logits.reshape(len(batch), -1)[:, 0])
        return np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, dtype=np.float32)


READER_DIR = _CACHE_DIR / "models" / "minilm-uncased-squad2-onnx"


class SquadReader:
    """deepset/minilm-uncased-squad2 (extractive QA trained on SQuAD 2.0, so
    it also scores "no answer"), converted to ONNX once by
    streaming_rag/retrieval/export_reader.py. Used only as the answerability
    signal of the refusal gate."""

    def __init__(self, folder: Path = READER_DIR, max_length: int = 384, max_answer_tokens: int = 30):
        from tokenizers import Tokenizer
        if not (folder / "model.onnx").exists():
            raise ModelNotAvailable(
                f"SQuAD2 reader not found in {folder}. Run: python -m streaming_rag.retrieval.export_reader")
        self._session = _session(str(folder / "model.onnx"))
        self._tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length, strategy="only_second")
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._max_answer = max_answer_tokens

    def margin(self, question: str, passages: list[str]) -> float | None:
        """Best answer-span score minus the lowest no-answer score over the
        passages (higher = more confident an answer is present)."""
        if not passages:
            return None
        encs = self._tok.encode_batch([(question, p) for p in passages])
        start, end = self._session.run(None, _feeds(self._session, encs))
        best, nulls = -1e9, []
        for i, enc in enumerate(encs):
            nulls.append(float(start[i][0] + end[i][0]))
            ctx = [j for j, sid in enumerate(enc.sequence_ids) if sid == 1]
            if not ctx:
                continue
            lo, hi = ctx[0], ctx[-1]
            top_s = np.argsort(-start[i][lo:hi + 1])[:20] + lo
            top_e = np.argsort(-end[i][lo:hi + 1])[:20] + lo
            for s in top_s:
                for e in top_e:
                    if s <= e < s + self._max_answer:
                        best = max(best, float(start[i][s] + end[i][e]))
        return None if best <= -1e9 else best - min(nulls)


def get_reader() -> SquadReader:
    return _get("reader", SquadReader)


def get_e5_encoder() -> E5Encoder:
    return _get("e5", E5Encoder)


def get_cross_encoder() -> CrossEncoder:
    return _get("ce", CrossEncoder)


def _get(key: str, factory):
    with _lock:
        if key not in _loaded:
            _loaded[key] = factory()
        return _loaded[key]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify() -> list[str]:
    """Checks every pinned model file against its recorded SHA-256; returns
    the list of problems (empty = all files present and byte-identical)."""
    problems = []
    for spec in (E5_SMALL_V2, MS_MARCO_MINILM_L6):
        for f, expected in spec.sha256:
            try:
                actual = _sha256(_resolve(spec, f))
            except ModelNotAvailable as e:
                problems.append(str(e))
                continue
            if actual != expected:
                problems.append(f"{spec.repo} {f}: sha256 {actual} != pinned {expected}")
    return problems


def fetch() -> None:
    from huggingface_hub import hf_hub_download
    for spec in (E5_SMALL_V2, MS_MARCO_MINILM_L6):
        for f in spec.files:
            path = hf_hub_download(spec.repo, f, revision=spec.revision)
            print(f"{spec.repo}@{spec.revision[:8]} {f} -> {path}")
    problems = verify()
    if problems:
        raise SystemExit("model verification failed:\n  " + "\n  ".join(problems))
    print("all model files verified against pinned SHA-256")


if __name__ == "__main__":
    import sys
    if "--fetch" in sys.argv:
        fetch()
    elif "--verify" in sys.argv:
        issues = verify()
        print("\n".join(issues) if issues else "all model files verified against pinned SHA-256")
        sys.exit(1 if issues else 0)
    else:
        print(__doc__)
