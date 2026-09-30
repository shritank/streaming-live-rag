"""Test doubles and fixture builders.

Task 3 is testable alone because everything it depends on is behind a contract:
the language model, retrieval and telemetry all have fakes here. None of these
are importable from `streaming_rag/` — fixtures never leak into engine code.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from streaming_rag.contracts import (
    Chunk,
    EvidenceChunk,
    LLMResponse,
    RetrievalResult,
    SubQuery,
)


# ---------------------------------------------------------------- telemetry
class RecordingTelemetry:
    """Captures emitted events so tests can assert on the trace."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def of(self, event: str) -> list[dict[str, Any]]:
        return [fields for name, fields in self.events if name == event]


# ---------------------------------------------------------------------- LLM
@dataclass
class FakeLLM:
    """Scriptable async language model.

    ``responses`` maps a purpose ("synthesis", "delta_decisions", …) to a list of
    payloads returned in order. A payload may be a dict (serialised to JSON), a
    raw string, or an Exception instance to raise.
    """

    responses: dict[str, list[Any]] = field(default_factory=dict)
    delay_s: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, *, purpose: str, system: str, prompt: str,
                       json_mode: bool = False, max_tokens: int = 512) -> LLMResponse:
        self.calls.append({"purpose": purpose, "system": system, "prompt": prompt,
                           "json_mode": json_mode, "max_tokens": max_tokens})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)

        queue = self.responses.get(purpose)
        if not queue:
            raise AssertionError(f"FakeLLM has no scripted response for {purpose!r}")
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(payload, Exception):
            raise payload
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(text=text, model="fake", tokens_in=len(prompt) // 4,
                           tokens_out=len(text) // 4, latency_ms=self.delay_s * 1000)

    def purposes(self) -> list[str]:
        return [call["purpose"] for call in self.calls]


# ---------------------------------------------------------------- retrieval
class FakeRetriever:
    """Records every search. Task 3 must never call it.

    The synthesizer takes no retriever at all, so this fake exists to prove the
    negative: refinement and restructuring cause no searches beyond the deltas
    the engine already performed.
    """

    def __init__(self, results: dict[str, RetrievalResult] | None = None) -> None:
        self.results = results or {}
        self.calls: list[list[str]] = []

    async def search(self, queries: list[SubQuery], k: int = 5) -> list[RetrievalResult]:
        self.calls.append([q.query_id for q in queries])
        return [self.results.get(q.query_id, empty_result(q.query_id)) for q in queries]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        for result in self.results.values():
            for ev in result.evidence:
                if ev.chunk.chunk_id == chunk_id:
                    return ev.chunk
        return None

    @property
    def call_count(self) -> int:
        return len(self.calls)


# ------------------------------------------------------------------ builders
def chunk(doc: str, section: str, text: str, n: int = 0, **metadata: Any) -> Chunk:
    return Chunk(chunk_id=f"{doc} §{section} #{n}", doc_id=doc, section=section,
                 text=text, metadata=metadata)


def evidence(chunks_and_scores: list[tuple[Chunk, float]],
             query_id: str) -> tuple[EvidenceChunk, ...]:
    return tuple(
        EvidenceChunk(chunk=c, score=s, query_ids=(query_id,), dense_rank=i, sparse_rank=i)
        for i, (c, s) in enumerate(chunks_and_scores)
    )


def result(query_id: str, chunks: list[Chunk], scores: list[float] | None = None,
           low_confidence: bool = False, latency_ms: float = 42.0) -> RetrievalResult:
    scores = scores or [1.0 - 0.1 * i for i in range(len(chunks))]
    return RetrievalResult(
        query_id=query_id,
        evidence=evidence(list(zip(chunks, scores, strict=False)), query_id),
        latency_ms=latency_ms,
        low_confidence=low_confidence,
    )


def empty_result(query_id: str) -> RetrievalResult:
    return RetrievalResult(query_id=query_id, evidence=(), latency_ms=5.0,
                           low_confidence=True)


def sub_query(query_id: str, text: str, label: str, trigger: str = "multi_intent",
              utterance: str = "u1", parent: str | None = None) -> SubQuery:
    return SubQuery(query_id=query_id, text=text, intent_label=label,
                    trigger=trigger, utterance_id=utterance, parent_query_id=parent)


def claim_payload(rows: list[tuple[str, list[str], str | None]]) -> dict[str, Any]:
    """Build the JSON a well-behaved model would return for synthesis."""
    return {"claims": [{"text": text, "citations": citations, "sub_query_id": query_id}
                       for text, citations, query_id in rows]}


def decisions_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"decisions": rows}
