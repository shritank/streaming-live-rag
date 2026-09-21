"""Dense retrieval side of the hybrid.

The default encoder is LSA (TF-IDF + truncated SVD). It is deterministic,
builds in about a second on a corpus of this size, needs no model download,
and — unlike BM25 — bridges vocabulary gaps via term co-occurrence
("refund" <-> "cancellation"), which is exactly what the hybrid-vs-sparse
ablation is meant to measure.

A transformer encoder can be swapped in without touching the retriever:
any object with `.encode(list[str]) -> np.ndarray` satisfies `Embedder`.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .text import content_tokens


class Embedder(Protocol):
    dim: int
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LsaEmbedder:
    def __init__(self, dim: int = 128, seed: int = 7):
        self.dim = dim
        self._seed = seed
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._components: np.ndarray | None = None   # (dim, vocab)
        self._doc_vectors: np.ndarray | None = None  # (n_docs, dim), L2-normalised

    def fit(self, documents: list[str]) -> "LsaEmbedder":
        tokenized = [content_tokens(d) for d in documents]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        n_docs, n_terms = len(documents), len(vocab)
        if n_docs == 0 or n_terms == 0:
            self._idf = np.zeros(0)
            self._components = np.zeros((0, 0))
            self._doc_vectors = np.zeros((0, 0))
            return self

        tf = np.zeros((n_docs, n_terms), dtype=np.float64)
        for i, tokens in enumerate(tokenized):
            for t in tokens:
                tf[i, vocab[t]] += 1.0
        df = (tf > 0).sum(axis=0)
        self._idf = np.log((n_docs + 1) / (df + 1)) + 1.0

        tfidf = np.log1p(tf) * self._idf
        tfidf = _l2_normalize(tfidf)

        k = min(self.dim, min(tfidf.shape) - 1) if min(tfidf.shape) > 1 else 1
        # Deterministic truncated SVD. full_matrices=False keeps this cheap at
        # corpus scale; the seed only matters if a randomized path is used.
        u, s, vt = np.linalg.svd(tfidf, full_matrices=False)
        self.dim = k
        self._components = vt[:k]                       # (k, n_terms)
        self._doc_vectors = _l2_normalize(u[:, :k] * s[:k])
        return self

    @property
    def doc_vectors(self) -> np.ndarray:
        if self._doc_vectors is None:
            raise RuntimeError("LsaEmbedder.fit() must be called before use")
        return self._doc_vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._components is None or self._idf is None:
            raise RuntimeError("LsaEmbedder.fit() must be called before use")
        if self._components.size == 0:
            return np.zeros((len(texts), 0))
        raw = np.zeros((len(texts), len(self._vocab)), dtype=np.float64)
        for i, text in enumerate(texts):
            for t in content_tokens(text):
                idx = self._vocab.get(t)
                if idx is not None:
                    raw[i, idx] += 1.0
        tfidf = _l2_normalize(np.log1p(raw) * self._idf)
        return _l2_normalize(tfidf @ self._components.T)


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def cosine_top_k(query_vecs: np.ndarray, doc_vecs: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
    """Both inputs are L2-normalised, so the dot product is cosine similarity."""
    if doc_vecs.size == 0 or query_vecs.size == 0:
        return [[] for _ in range(len(query_vecs))]
    sims = query_vecs @ doc_vecs.T
    out = []
    for row in sims:
        order = np.argsort(-row, kind="stable")[:k]
        out.append([(int(i), float(row[i])) for i in order if row[i] > 0])
    return out
