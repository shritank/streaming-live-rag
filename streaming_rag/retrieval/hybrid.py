"""HybridRetriever — the Retriever implementation (Task 1).

search() runs every sub-query concurrently. The CPU-bound scoring happens in
a worker thread via asyncio.to_thread, so a retrieval never blocks the event
loop while the user is still speaking (project context §9.2).
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..contracts import Chunk, EvidenceChunk, RetrievalResult, SubQuery, Telemetry, NullTelemetry
from .bm25 import BM25Index
from .embed import LsaEmbedder, cosine_top_k
from .fusion import rrf
from .ingest import load_corpus
from .rerank import dedup, rerank


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

    async def setup(self) -> None:
        """Cold start: ingest, index and fit the encoder. Off the per-turn clock."""
        if self._ready:
            return
        if not self._chunks:
            self._chunks = await asyncio.to_thread(load_corpus, self._config.corpus_dir)
        await asyncio.to_thread(self._build_indexes)
        self._ready = True

    def _build_indexes(self) -> None:
        texts = [self._searchable_text(c) for c in self._chunks]
        self._indexed_texts = texts
        self._bm25 = BM25Index(texts)
        self._embedder = LsaEmbedder().fit(texts)
        self._by_id = {c.chunk_id: c for c in self._chunks}
        self._by_citation = {}
        for c in self._chunks:
            self._by_citation.setdefault(c.citation, c)

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        heading = chunk.metadata.get("heading", "")
        title = chunk.metadata.get("doc_title", "")
        return f"{title} {heading} {chunk.text}".strip()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def get_chunk_by_citation(self, citation: str) -> Chunk | None:
        """Used by the grounding verifier to prove a cited `Doc_ID §Section`
        actually exists in the corpus (G4: zero fabricated document IDs)."""
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
        start = time.monotonic()
        evidence = await asyncio.to_thread(self._search_sync, query, k)
        extra = self._config.retrieval.simulated_latency_ms
        if extra:
            await asyncio.sleep(extra / 1000)
        latency_ms = (time.monotonic() - start) * 1000
        best = evidence[0].score if evidence else 0.0
        return RetrievalResult(
            query_id=query.query_id,
            evidence=tuple(evidence),
            latency_ms=latency_ms,
            low_confidence=best < self._config.retrieval.low_confidence_threshold,
        )

    def _search_sync(self, query: SubQuery, k: int) -> list[EvidenceChunk]:
        assert self._bm25 is not None and self._embedder is not None
        mode = self._config.retrieval.mode
        pool = max(k * 4, 20)

        sparse_ranked: list[int] = []
        dense_ranked: list[int] = []

        if mode in ("hybrid", "sparse"):
            sparse_ranked = [i for i, _ in self._bm25.top_k(query.text, pool)]
        if mode in ("hybrid", "dense"):
            q_vec = self._embedder.encode([query.text])
            dense_ranked = [i for i, _ in cosine_top_k(q_vec, self._embedder.doc_vectors, pool)[0]]

        rank_lists = {}
        if sparse_ranked:
            rank_lists["sparse"] = sparse_ranked
        if dense_ranked:
            rank_lists["dense"] = dense_ranked
        if not rank_lists:
            return []

        fused = rrf(rank_lists, rrf_k=self._config.retrieval.rrf_k, weights={
            "sparse": self._config.retrieval.sparse_weight,
            "dense": self._config.retrieval.dense_weight,
        })

        sparse_pos = {doc: r for r, doc in enumerate(sparse_ranked)}
        dense_pos = {doc: r for r, doc in enumerate(dense_ranked)}

        # Normalise fused scores to [0,1] before reranking: raw RRF values are
        # ~1/rrf_k, which would otherwise be swamped by the rerank features and
        # make the retrieval mode irrelevant to the final ordering.
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

    def _index_of(self, chunk: Chunk) -> int:
        # chunk_id is unique, and self._chunks order is stable after setup.
        return self._chunk_index.get(chunk.chunk_id, -1)

    @property
    def _chunk_index(self) -> dict[str, int]:
        if not hasattr(self, "_chunk_index_cache") or self._chunk_index_cache_len != len(self._chunks):
            self._chunk_index_cache = {c.chunk_id: i for i, c in enumerate(self._chunks)}
            self._chunk_index_cache_len = len(self._chunks)
        return self._chunk_index_cache
