"""HybridRetriever — the Retriever implementation (Task 1).

search() runs every sub-query concurrently. The CPU-bound scoring happens in
a worker thread via asyncio.to_thread, so a retrieval never blocks the event
loop while the user is still speaking (project context §9.2).
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..contracts import Chunk, EvidenceChunk, RetrievalResult, SubQuery, Telemetry, NullTelemetry
from .bm25 import BM25Index
from .embed import LsaEmbedder, cosine_top_k
from .fusion import rrf
from .ingest import load_corpus
from .rerank import dedup, rerank


@dataclass
class _IndexBundle:
    chunks: list[Chunk]
    texts: list[str]
    bm25: BM25Index
    embedder: object
    specific_vocab: frozenset[str]
    low_df_bigrams: frozenset[str]
    char_index: object = None


class _CharNgramIndex:
    """TF-IDF over character 3-5-grams inside word boundaries (sklearn)."""

    def __init__(self, texts: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self._matrix = self._vec.fit_transform(texts)

    def top_k(self, query: str, k: int) -> list[int]:
        sims = (self._matrix @ self._vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-sims, kind="stable")[:k]
        return [int(i) for i in order if sims[i] > 0]


# Read-only indexes shared by every retriever in the process that serves the
# same corpus with the same dense encoder (e.g. one index, many sessions).
# Keyed on the corpus files' paths, sizes and mtimes so an edited corpus is
# re-indexed rather than served stale.
_INDEX_CACHE: dict[tuple, _IndexBundle] = {}
_INDEX_LOCK = threading.Lock()


def _corpus_fingerprint(corpus_dir: str) -> tuple:
    root = Path(corpus_dir).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return (str(root),) + tuple((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in files)


def _sections_by_citation(chunks: list[Chunk]) -> dict[str, Chunk]:
    grouped: dict[str, list[Chunk]] = {}
    for c in chunks:
        grouped.setdefault(c.citation, []).append(c)
    return {cit: Chunk(chunk_id=parts[0].chunk_id, doc_id=parts[0].doc_id, section=parts[0].section,
                       text=" ".join(p.text for p in parts), metadata=dict(parts[0].metadata))
            for cit, parts in grouped.items()}


def _make_embedder(name: str):
    if name == "lsa":
        return LsaEmbedder()
    if name == "e5-small-v2":
        from .neural import E5Embedder
        return E5Embedder()
    raise ValueError(f"unknown retrieval.dense_encoder {name!r} (lsa | e5-small-v2)")


class HybridRetriever:
    def __init__(self, config: Config, telemetry: Telemetry | None = None,
                 chunks: list[Chunk] | None = None):
        self._config = config
        self._telemetry = telemetry or NullTelemetry()
        self._chunks: list[Chunk] = chunks or []
        self._indexed_texts: list[str] = []
        self._by_id: dict[str, Chunk] = {}
        self._by_citation: dict[str, Chunk] = {}
        self._bm25: BM25Index | None = None
        self._embedder: LsaEmbedder | None = None
        self._ready = False
        self._specific_vocab: frozenset[str] = frozenset()
        self._low_df_bigrams: frozenset[str] = frozenset()
        self._cross_encoder = None
        self._char_index = None

    async def setup(self) -> None:
        """Cold start: ingest, index and fit the encoder. Off the per-turn clock."""
        if self._ready:
            return
        if self._chunks:
            await asyncio.to_thread(self._build_indexes)
        else:
            await asyncio.to_thread(self._load_shared_indexes)
        if self._config.retrieval.reranker == "cross-encoder":
            from .neural import get_cross_encoder
            self._cross_encoder = await asyncio.to_thread(get_cross_encoder)
        self._ready = True

    def _load_shared_indexes(self) -> None:
        key = (_corpus_fingerprint(self._config.corpus_dir), self._config.retrieval.dense_encoder,
               self._config.retrieval.chunk_chars, self._config.retrieval.char_weight > 0)
        with _INDEX_LOCK:
            bundle = _INDEX_CACHE.get(key)
            if bundle is None:
                self._chunks = load_corpus(self._config.corpus_dir, max_chars=self._config.retrieval.chunk_chars)
                bundle = self._index(self._chunks)
                _INDEX_CACHE[key] = bundle
        self._install(bundle)

    def _build_indexes(self) -> None:
        self._install(self._index(self._chunks))

    def _index(self, chunks: list[Chunk]) -> _IndexBundle:
        texts = [self._searchable_text(c) for c in chunks]
        bm25 = BM25Index(texts)
        embedder = _make_embedder(self._config.retrieval.dense_encoder).fit(texts)
        char_index = _CharNgramIndex(texts) if self._config.retrieval.char_weight > 0 else None
        return _IndexBundle(chunks=list(chunks), texts=texts, bm25=bm25, embedder=embedder,
                            specific_vocab=bm25.specific_terms(), low_df_bigrams=bm25.low_df_bigrams(),
                            char_index=char_index)

    def _install(self, bundle: _IndexBundle) -> None:
        self._chunks = list(bundle.chunks)
        self._indexed_texts = bundle.texts
        self._bm25 = bundle.bm25
        self._specific_vocab = bundle.specific_vocab
        self._low_df_bigrams = bundle.low_df_bigrams
        self._embedder = bundle.embedder
        self._char_index = bundle.char_index
        self._by_id = {c.chunk_id: c for c in self._chunks}
        self._by_citation = _sections_by_citation(self._chunks)

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        heading = chunk.metadata.get("heading", "")
        title = chunk.metadata.get("doc_title", "")
        return f"{title} {heading} {chunk.text}".strip()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    @property
    def specific_vocabulary(self) -> frozenset[str]:
        """Discriminative corpus terms (see BM25Index.specific_terms), exposed
        for the controller's weak-anchor heuristic. Empty until setup()."""
        return self._specific_vocab

    @property
    def low_df_bigrams(self) -> frozenset[str]:
        """Low document-frequency content-token bigrams (see
        BM25Index.low_df_bigrams), exposed for the controller's weak-anchor
        heuristic. Empty until setup()."""
        return self._low_df_bigrams

    def get_chunk_by_citation(self, citation: str) -> Chunk | None:
        """Used by the grounding verifier to prove a cited `Doc_ID §Section`
        actually exists in the corpus (G4: zero fabricated document IDs) and
        to check a claim against it. Returns the WHOLE section: a citation
        names a section, and a long section is split over several chunks, so
        resolving to only its first chunk made claims drawn from a later chunk
        look unsupported."""
        return self._by_citation.get(citation)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    async def search(self, queries: list[SubQuery], k: int = 5) -> list[RetrievalResult]:
        if not self._ready:
            await self.setup()
        if not queries:
            return []
        return await asyncio.gather(*(self._search_one(q, k) for q in queries))

    async def _search_one(self, query: SubQuery, k: int) -> RetrievalResult:
        start = time.perf_counter()
        evidence = await asyncio.to_thread(self._search_sync, query, k)
        extra = self._config.retrieval.simulated_latency_ms
        if extra:
            await asyncio.sleep(extra / 1000)
        latency_ms = (time.perf_counter() - start) * 1000
        best = evidence[0].score if evidence else float("-inf")
        threshold = (self._config.retrieval.ce_low_confidence_logit
                     if self._cross_encoder is not None
                     else self._config.retrieval.low_confidence_threshold)
        return RetrievalResult(
            query_id=query.query_id,
            evidence=tuple(evidence),
            latency_ms=latency_ms,
            low_confidence=best < threshold,
        )

    def _search_sync(self, query: SubQuery, k: int) -> list[EvidenceChunk]:
        assert self._bm25 is not None and self._embedder is not None
        mode = self._config.retrieval.mode
        pool = max(k * 4, 20)

        sparse_ranked: list[int] = []
        dense_ranked: list[int] = []

        if mode in ("hybrid", "sparse"):
            text = self._prf_query(query.text) if self._config.retrieval.bm25_prf else query.text
            sparse_ranked = [i for i, _ in self._bm25.top_k(text, pool)]
        if mode in ("hybrid", "dense"):
            q_vec = self._embedder.encode([query.text])
            dense_ranked = [i for i, _ in cosine_top_k(q_vec, self._embedder.doc_vectors, pool)[0]]

        rank_lists = {}
        if sparse_ranked:
            rank_lists["sparse"] = sparse_ranked
        if dense_ranked:
            rank_lists["dense"] = dense_ranked
        if self._char_index is not None and mode == "hybrid":
            char_ranked = self._char_index.top_k(query.text, pool)
            if char_ranked:
                rank_lists["char"] = char_ranked
        if not rank_lists:
            return []

        fused = rrf(rank_lists, rrf_k=self._config.retrieval.rrf_k, weights={
            "sparse": self._config.retrieval.sparse_weight,
            "dense": self._config.retrieval.dense_weight,
            "char": self._config.retrieval.char_weight,
        })

        sparse_pos = {doc: r for r, doc in enumerate(sparse_ranked)}
        dense_pos = {doc: r for r, doc in enumerate(dense_ranked)}

        if self._cross_encoder is not None:
            order = sorted(fused, key=lambda i: (-fused[i], i))
            pool_ids, tail_ids = order[:self._config.retrieval.rerank_pool], order[self._config.retrieval.rerank_pool:]
            logits = self._cross_encoder.score(query.text, [self._indexed_texts[i] for i in pool_ids])
            ranked = sorted(((self._chunks[i], float(s)) for i, s in zip(pool_ids, logits)),
                            key=lambda t: (-t[1], t[0].chunk_id))
            # Candidates past the reranked pool keep their fused order, ranked
            # below every reranked one, so a small pool never shrinks recall@k.
            floor = min((s for _, s in ranked), default=0.0) - 1.0
            ranked += [(self._chunks[i], floor) for i in tail_ids]
        elif self._config.retrieval.reranker == "none":
            max_fused = max(fused.values()) if fused else 1.0
            ranked = sorted(((self._chunks[i], s / max_fused) for i, s in fused.items()),
                            key=lambda t: (-t[1], t[0].chunk_id))
        else:
            # Normalise fused scores to [0,1] before reranking: raw RRF values
            # are ~1/rrf_k, which would otherwise be swamped by the rerank
            # features and make the retrieval mode irrelevant to the ordering.
            max_fused = max(fused.values()) if fused else 1.0
            candidates = [(self._chunks[i], score / max_fused, self._indexed_texts[i])
                           for i, score in fused.items()]
            ranked = rerank(query.text, candidates)
        deduped = dedup(ranked)[:k]

        return [
            EvidenceChunk(
                chunk=chunk,
                score=score,
                query_ids=(query.query_id,),
                dense_rank=dense_pos.get(self._index_of(chunk)),
                sparse_rank=sparse_pos.get(self._index_of(chunk)),
            )
            for chunk, score in deduped
        ]

    def _prf_query(self, query: str, fb_docs: int = 3, fb_terms: int = 5) -> str:
        """RM3-style expansion: the original terms (doubled, so they keep more
        weight in BM25's per-occurrence sum) plus the highest-weighted new
        terms of the first-pass top documents."""
        from collections import Counter
        from .text import content_tokens
        q_terms = content_tokens(query)
        top = self._bm25.top_k(query, fb_docs)
        weights: Counter = Counter()
        for i, score in top:
            tokens = content_tokens(self._indexed_texts[i])
            for t, c in Counter(tokens).items():
                weights[t] += score * c / len(tokens) * self._bm25._idf.get(t, 0.0)
        extra = [t for t, _ in weights.most_common() if t not in q_terms][:fb_terms]
        return " ".join(q_terms * 2 + extra)

    def _index_of(self, chunk: Chunk) -> int:
        # chunk_id is unique, and self._chunks order is stable after setup.
        return self._chunk_index.get(chunk.chunk_id, -1)

    @property
    def _chunk_index(self) -> dict[str, int]:
        if not hasattr(self, "_chunk_index_cache") or self._chunk_index_cache_len != len(self._chunks):
            self._chunk_index_cache = {c.chunk_id: i for i, c in enumerate(self._chunks)}
            self._chunk_index_cache_len = len(self._chunks)
        return self._chunk_index_cache
