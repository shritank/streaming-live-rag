"""Experiment-only model swaps (production code is not modified).

    python -m eval.model_zoo --ce minilm-l12 --embedder bge-small -- retrieval_eval --corpus-dir ...
    python -m eval.model_zoo --ce tinybert-l2 -- run_quality --scenarios ... --corpus-dir ...

Loads candidate models from an isolated folder (STREAMING_RAG_ZOO, default
the session scratch 'iso/models') and injects them into this process only:
the cross-encoder singleton in streaming_rag.retrieval.neural and the dense
encoder factory in streaming_rag.retrieval.hybrid. Everything else — the
engine, scoring and statistics — is the unchanged production path, so a
difference in results is attributable to the swapped model alone.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ZOO = Path(os.environ.get("STREAMING_RAG_ZOO",
                          "E:/WindowsTemp/claude/E--Downloads-SamsungGenAI--1-/c102343b-1cac-44a8-b83b-017d6095c925/scratchpad/iso/models"))

CROSS_ENCODERS = {
    "minilm-l12": "cross-encoder--ms-marco-MiniLM-L12-v2",
    "tinybert-l2": "cross-encoder--ms-marco-TinyBERT-L2-v2",
}


class PathCrossEncoder:
    """Same inference code as neural.CrossEncoder, loaded from a folder."""

    def __init__(self, folder: Path, batch_size: int = 32, max_length: int = 512):
        from streaming_rag.retrieval import neural
        self._session = neural._session(str(folder / "onnx" / "model.onnx"))
        self._tok = neural._tokenizer(str(folder / "tokenizer.json"), max_length)
        self._batch_size = batch_size
        self._feeds = neural._feeds

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(passages), self._batch_size):
            batch = self._tok.encode_batch([(query, p) for p in passages[start:start + self._batch_size]])
            logits = self._session.run(None, self._feeds(self._session, batch))[0]
            out.append(logits.reshape(len(batch), -1)[:, 0])
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, dtype=np.float32)


class BgeSmallEmbedder:
    """BAAI/bge-small-en-v1.5: CLS pooling, L2-normalised, retrieval queries
    prefixed with the instruction from its model card; passages unprefixed."""
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, folder: Path | None = None, batch_size: int = 32):
        from streaming_rag.retrieval import neural
        folder = folder or ZOO / "BAAI--bge-small-en-v1.5"
        self._session = neural._session(str(folder / "onnx" / "model.onnx"))
        self._tok = neural._tokenizer(str(folder / "tokenizer.json"), 512)
        self._feeds = neural._feeds
        self._batch_size = batch_size
        self.dim = 384
        self._doc_vectors = None

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self._batch_size):
            batch = self._tok.encode_batch(texts[start:start + self._batch_size])
            hidden = self._session.run(None, self._feeds(self._session, batch))[0]
            out.append(hidden[:, 0, :])
        vecs = np.concatenate(out) if out else np.zeros((0, self.dim), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vecs / norms).astype(np.float32)

    def fit(self, documents: list[str]) -> "BgeSmallEmbedder":
        import hashlib
        digest = hashlib.sha256("\x1e".join(documents).encode("utf-8")).hexdigest()[:24]
        cache = Path(".cache/streaming_rag/embeddings/bge-small-en-v1.5@5c38ec7c") / f"{digest}.npy"
        if cache.exists():
            vecs = np.load(cache)
            if vecs.shape == (len(documents), self.dim):
                self._doc_vectors = vecs
                return self
        self._doc_vectors = self._encode(documents)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, self._doc_vectors)
        return self

    @property
    def doc_vectors(self) -> np.ndarray:
        return self._doc_vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.QUERY_PREFIX + t for t in texts])


class SquadReader:
    """deepset/minilm-uncased-squad2 exported to ONNX (iso/export_reader.py).
    Returns, for a question and passages, the best answer span per passage
    with its score and the model's no-answer (CLS) score."""

    def __init__(self, folder: Path | None = None, max_length: int = 384, max_answer_tokens: int = 30):
        from streaming_rag.retrieval import neural
        from tokenizers import Tokenizer
        folder = folder or ZOO / "deepset--minilm-uncased-squad2-onnx"
        self._session = neural._session(str(folder / "model.onnx"))
        self._tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length, strategy="only_second")
        self._tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self._feeds = neural._feeds
        self._max_answer = max_answer_tokens

    def read(self, question: str, passages: list[str]) -> list[dict]:
        if not passages:
            return []
        encs = self._tok.encode_batch([(question, p) for p in passages])
        start, end = self._session.run(None, self._feeds(self._session, encs))
        out = []
        for i, enc in enumerate(encs):
            ctx = [j for j, sid in enumerate(enc.sequence_ids) if sid == 1]
            null = float(start[i][0] + end[i][0])
            best, span = -1e9, None
            if ctx:
                lo, hi = ctx[0], ctx[-1]
                s_log, e_log = start[i], end[i]
                top_s = np.argsort(-s_log[lo:hi + 1])[:20] + lo
                top_e = np.argsort(-e_log[lo:hi + 1])[:20] + lo
                for s in top_s:
                    for e in top_e:
                        if s <= e < s + self._max_answer and s_log[s] + e_log[e] > best:
                            best, span = float(s_log[s] + e_log[e]), (int(s), int(e))
            if span is None:
                out.append({"score": -1e9, "null": null, "text": "", "char_start": -1, "char_end": -1})
                continue
            cs, ce = enc.offsets[span[0]][0], enc.offsets[span[1]][1]
            out.append({"score": best, "null": null, "text": passages[i][cs:ce], "char_start": cs, "char_end": ce})
        return out


READER_MARGINS: dict[str, float] = {}


def _reader_claim(self, query, result):
    """Replacement for GroundedSynthesizer._ce_claim in reader experiments:
    read the top chunks, take the passage whose best span beats the others,
    and assert the SENTENCE containing that span (claims stay verbatim,
    citable sentences). The span-vs-no-answer margin is recorded for refusal
    analysis; the claim is withheld when the margin is below
    synthesis.ce_claim_threshold."""
    from streaming_rag.contracts import Claim
    from streaming_rag.retrieval.text import split_sentences
    evidence = list(result.evidence[:self._config.synthesis.ce_evidence_chunks])
    if not evidence:
        return None
    reads = _reader().read(query, [e.chunk.text for e in evidence])
    best_i = max(range(len(reads)), key=lambda i: reads[i]["score"])
    r = reads[best_i]
    if r["char_start"] < 0:
        return None
    margin = r["score"] - min(x["null"] for x in reads)
    text = evidence[best_i].chunk.text
    pos, sentence = 0, text
    for s in split_sentences(text):
        k = text.find(s, pos)
        if k <= r["char_start"] < k + len(s) + 1:
            sentence = s.strip()
            break
        pos = max(pos, k)
    READER_MARGINS[sentence] = margin
    if margin < self._config.synthesis.ce_claim_threshold:
        return None
    return Claim(text=sentence, citations=[evidence[best_i].chunk.citation])


_READER = None


def _reader() -> SquadReader:
    global _READER
    if _READER is None:
        _READER = SquadReader()
    return _READER


def install(ce: str | None = None, embedder: str | None = None, reader: bool = False) -> None:
    if reader:
        from streaming_rag.session.synthesis import GroundedSynthesizer
        GroundedSynthesizer._ce_claim_original = GroundedSynthesizer._ce_claim
        GroundedSynthesizer._ce_claim = _reader_claim
    from streaming_rag.retrieval import hybrid, neural
    if ce:
        model = PathCrossEncoder(ZOO / CROSS_ENCODERS[ce])
        neural._loaded["ce"] = model
    if embedder == "bge-small":
        original = hybrid._make_embedder

        def make(name: str):
            return BgeSmallEmbedder() if name == "e5-small-v2" else original(name)
        hybrid._make_embedder = make
        # a bge index must not be served from an e5 cache entry
        hybrid._INDEX_CACHE.clear()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--")
    opts, rest = argv[:split], argv[split + 1:]
    ce = opts[opts.index("--ce") + 1] if "--ce" in opts else None
    embedder = opts[opts.index("--embedder") + 1] if "--embedder" in opts else None
    install(ce, embedder, reader="--reader" in opts)
    command, args = rest[0], rest[1:]
    if command == "retrieval_eval":
        from .retrieval_eval import main as run
    elif command == "run_quality":
        from .run_quality import main as run
    elif command == "refusal_sweep":
        from .refusal_sweep import main as run
    else:
        raise SystemExit(f"unknown command {command}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
