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
import logging
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

_log = logging.getLogger("streaming_rag.neural")
ACTIVE_PROVIDERS: dict[str, str] = {}   # model file -> execution provider its session bound
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


def _session(path: str, max_len: int = 512):
    import onnxruntime as ort

    from ..gpu import check_ort_session, ort_use_cuda
    cuda = ort_use_cuda()
    # auto (default): CUDA, mandatory whenever an NVIDIA GPU is present
    # (streaming_rag/gpu.py); cpu only as an explicit opt-out. The CPU EP stays
    # listed only for the few int64 mask/shape nodes CUDA has no kernel for;
    # the session must still bind CUDA first.
    opts = ort.SessionOptions()
    # CUDA keeps the measured 4. On the CPU the model calls ARE the latency, so use half the logical cores
    # (about the physical cores), between 4 and 8, unless STREAMING_RAG_ORT_THREADS says otherwise
    # (CPU micro-benchmark, 20-passage rerank: 823 ms at 4 threads, 550 ms at 8).
    cpu_threads = max(4, min(8, (os.cpu_count() or 8) // 2))
    opts.intra_op_num_threads = int(os.environ.get("STREAMING_RAG_ORT_THREADS") or ("4" if cuda else str(cpu_threads)))
    opts.inter_op_num_threads = 1
    # Sessions per model (each its own CUDA stream). Default 1: a pool of 3
    # halves per-call latency only at >= 4 concurrent callers (micro-benchmark);
    # the real pipeline runs ~2-way and had WORSE tails with 3 (rerank p99
    # 41-52 vs 26 ms, docs/experiment_log.md W6) - kept only as a knob.
    size = int(os.environ.get("STREAMING_RAG_ORT_SESSIONS", "1"))
    sessions = []
    for _ in range(max(1, size)):
        if cuda:
            s = ort.InferenceSession(path, sess_options=opts,
                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            check_ort_session(s, path)
        else:
            s = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        _warm_up(s, max_len)
        sessions.append(s)
    pool = _SessionPool(sessions)
    ACTIVE_PROVIDERS[path] = pool.get_providers()[0]
    _log.info("%s -> %s x%d", path, ACTIVE_PROVIDERS[path], len(sessions))
    return pool


class _SessionPool:
    """Same interface as an InferenceSession; each run() checks out an idle
    session (blocking if all are busy), so concurrent callers use separate
    sessions. Identical weights and graph: which session runs a call cannot
    change its result."""

    def __init__(self, sessions):
        import queue
        self._sessions = sessions
        self._idle = queue.SimpleQueue()
        for s in sessions:
            self._idle.put(s)

    def run(self, output_names, feeds):
        s = self._idle.get()
        try:
            return s.run(output_names, feeds)
        finally:
            self._idle.put(s)

    def get_inputs(self):
        return self._sessions[0].get_inputs()

    def get_providers(self):
        return self._sessions[0].get_providers()

    def __len__(self):
        return len(self._sessions)


def _warm_up(session, max_len: int) -> None:
    """Run the model once per representative shape at load time, so the first
    user question does not pay one-off initialisation (CUDA: cuBLAS/cuDNN
    handles, kernel loading, memory-arena growth; measured 100-540 ms spikes
    on the first retrieving turn). Stateless: no effect on any output."""
    names = {i.name for i in session.get_inputs()}
    for batch, seq in ((1, 16), (5, 192), (12, 96), (3, max_len)):
        feeds = {"input_ids": np.full((batch, seq), 1000, np.int64),
                 "attention_mask": np.ones((batch, seq), np.int64)}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros((batch, seq), np.int64)
        session.run(None, feeds)


FUSED_DIR = _CACHE_DIR / "models" / "fused"


def _onnx_path(key: str, original: str) -> str:
    """On CUDA (STREAMING_RAG_ORT_FUSED=auto, the default, or 1): the same
    weights with ONNX Runtime's transformer fusions (Attention/SkipLayerNorm/
    BiasGelu, fp32), built by `python -m streaming_rag.retrieval.neural
    --fuse`. Far fewer kernel launches per call; answers bit-identical on 618
    dev scenarios (docs/experiment_log.md, GPU pass). 0 = original graphs.

    On the CPU, STREAMING_RAG_ORT_QUANT=int8 (explicit opt-in) selects the dynamically int8-quantised
    cross-encoder / reader built by `python -m streaming_rag.retrieval.neural --quantize`."""
    from ..gpu import ort_use_cuda
    quant = os.environ.get("STREAMING_RAG_ORT_QUANT", "")
    if quant.startswith("int8") and not ort_use_cuda():
        # "int8" quantises both slow models; "int8:reader" / "int8:ce" just that one
        which = quant.split(":", 1)[1].split(",") if ":" in quant else ["ce", "reader"]
        if key in which:
            q = INT8_DIR / f"{key}.onnx"
            if not q.exists():
                raise ModelNotAvailable(f"{q} missing. Run: python -m streaming_rag.retrieval.neural --quantize")
            return str(q)
    if os.environ.get("STREAMING_RAG_ORT_FUSED", "auto") == "0":
        return original
    if not ort_use_cuda():
        return original
    fused = FUSED_DIR / f"{key}.onnx"
    if not fused.exists():
        if os.environ.get("STREAMING_RAG_ORT_FUSED", "auto") == "1":       # explicitly required
            raise ModelNotAvailable(f"{fused} missing. Run: python -m streaming_rag.retrieval.neural --fuse")
        # auto: still the GPU, just the original (slower, ~2x per call) graph - never the CPU
        _log.warning("fused CUDA graph %s not built - running the original graph on the GPU "
                     "(python -m streaming_rag.retrieval.neural --fuse builds it)", fused)
        return original
    return str(fused)


def fuse() -> None:
    """Build the CUDA graphs: cross-encoder and reader first get the exact
    static-shape rewrite (graph_static.staticize: bit-identical to the pinned
    export, moves their per-call shape arithmetic off the CPU EP), then all
    three get ONNX Runtime's transformer fusions. Writes FUSED_DIR/manifest.json."""
    import json

    import onnx
    from onnxruntime.transformers import optimizer

    from .graph_static import staticize
    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, src in (("e5", _resolve(E5_SMALL_V2, E5_SMALL_V2.onnx_file)),
                     ("ce", _resolve(MS_MARCO_MINILM_L6, MS_MARCO_MINILM_L6.onnx_file)),
                     ("reader", str(READER_DIR / "model.onnx"))):
        entry = {"source": str(src), "source_sha256": _sha256(src)}
        stage = src
        if key != "e5":   # e5's export has no run-time shape math (100% CUDA already)
            model, entry["staticize"] = staticize(Path(src).read_bytes())
            stage = str(FUSED_DIR / f"{key}.static.onnx")
            onnx.save(model, stage)
        opt = optimizer.optimize_model(stage, model_type="bert", num_heads=12, hidden_size=384,
                                       use_gpu=True, opt_level=0)
        dst = FUSED_DIR / f"{key}.onnx"
        opt.save_model_to_file(str(dst))
        if stage != src:
            Path(stage).unlink()
        entry["sha256"] = _sha256(str(dst))
        entry["fused_ops"] = {k: v for k, v in opt.get_fused_operator_statistics().items() if v}
        manifest[key] = entry
        print(f"{key}: {src} -> {dst} sha256 {entry['sha256'][:16]} {entry.get('staticize', '')}")
    (FUSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


INT8_DIR = _CACHE_DIR / "models" / "int8"


def quantize() -> None:
    """Build the CPU int8 models: dynamic weight-only int8 quantisation of the cross-encoder and the SQuAD2
    reader (the two slow models on a CPU; e5 stays fp32 so corpus embeddings are unchanged). Used only when
    STREAMING_RAG_ORT_QUANT=int8 and no GPU is in use. Writes INT8_DIR/manifest.json with the source and result
    hashes and sizes."""
    import json

    from onnxruntime.quantization import QuantType, quantize_dynamic
    INT8_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, src in (("ce", _resolve(MS_MARCO_MINILM_L6, MS_MARCO_MINILM_L6.onnx_file)),
                     ("reader", str(READER_DIR / "model.onnx"))):
        dst = INT8_DIR / f"{key}.onnx"
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
        manifest[key] = {"source": str(src), "source_sha256": _sha256(str(src)), "sha256": _sha256(str(dst)),
                         "source_bytes": Path(src).stat().st_size, "int8_bytes": dst.stat().st_size}
        print(f"{key}: {manifest[key]['source_bytes'] / 1e6:.1f} MB -> {manifest[key]['int8_bytes'] / 1e6:.1f} MB  {dst}")
    (INT8_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


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
        original = _resolve(spec, spec.onnx_file)
        path = _onnx_path("e5", original) if spec == E5_SMALL_V2 else original
        self.variant = "" if path == original else "-fused"   # separate corpus-vector cache
        self._session = _session(path)
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
        variant = getattr(self._encoder, "variant", "")
        cache = _CACHE_DIR / "embeddings" / f"e5-small-v2@{rev}{variant}" / f"{digest}.npy"
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
        original = _resolve(spec, spec.onnx_file)
        self._session = _session(_onnx_path("ce", original) if spec == MS_MARCO_MINILM_L6 else original)
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
        original = str(folder / "model.onnx")
        self._session = _session(_onnx_path("reader", original) if folder == READER_DIR else original,
                                 max_len=max_length)
        self._tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length, strategy="only_second")
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._max_answer = max_answer_tokens

    def margin(self, question: str, passages: list[str]) -> float | None:
        """Best answer-span score minus the lowest no-answer score over the
        passages (higher = more confident an answer is present)."""
        return self.answer(question, passages)[0]

    def answer(self, question: str, passages: list[str]) -> tuple[float | None, str]:
        """(margin, text of the single best answer span over all passages)."""
        if not passages:
            return None, ""
        encs = self._tok.encode_batch([(question, p) for p in passages])
        start, end = self._session.run(None, _feeds(self._session, encs))
        best, where, nulls = -1e9, None, []
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
                    if s <= e < s + self._max_answer and float(start[i][s] + end[i][e]) > best:
                        best, where = float(start[i][s] + end[i][e]), (i, int(s), int(e))
        if where is None:
            return None, ""
        i, s, e = where
        span = passages[i][encs[i].offsets[s][0]:encs[i].offsets[e][1]]
        return best - min(nulls), span


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
    elif "--fuse" in sys.argv:
        fuse()
    elif "--quantize" in sys.argv:
        quantize()
    elif "--verify" in sys.argv:
        issues = verify()
        print("\n".join(issues) if issues else "all model files verified against pinned SHA-256")
        sys.exit(1 if issues else 0)
    else:
        print(__doc__)
