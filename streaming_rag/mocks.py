"""Deterministic, contract-true mock components.

These let the engine, telemetry and eval harness be built and fully tested
before Tasks 1–3 land. Swapping a mock for the real component is a config
change: whichever object satisfies the Retriever / Synthesizer protocol.

Every mock supports fault injection (`fail_mode`) so the engine's
robustness path (tests §4.7) can be exercised without touching real code.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Literal

from .contracts import (
    Chunk, ControllerDecision, Decision, EvidenceChunk, RetrievalResult,
    SubQuery, TranscriptChunk, TurnKind, AnswerVersion, Claim, SessionView,
)

FailMode = Literal["none", "raise", "timeout", "empty", "malformed"]

_STOPWORDS = {"the", "a", "an", "is", "are", "for", "to", "and", "of", "in", "on", "please"}
_PRESENTATION_VERBS = {"repeat", "summarize", "shorten", "translate", "rephrase", "bullet", "bullets"}
_CHITCHAT = {"hi", "hello", "thanks", "thank you", "bye", "goodbye", "ok", "okay"}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class MockController:
    """Rule-based mock: WAIT until >= min_words, RETRIEVE with naive intent
    splitting on ' and ' / ',', SUPPRESS on presentation-only phrasing.
    Deterministic; no LLM calls unless fail_mode requests one to fail.
    """

    def __init__(self, llm=None, telemetry=None, config: dict | None = None, fail_mode: FailMode = "none"):
        self._llm = llm
        self._telemetry = telemetry
        self._config = config or {}
        self._fail_mode = fail_mode
        self._seen_by_utterance: dict[str, set[str]] = {}
        self._accumulated: dict[str, str] = {}

    def reset_utterance(self, utterance_id: str) -> None:
        self._seen_by_utterance.pop(utterance_id, None)
        self._accumulated.pop(utterance_id, None)

    async def on_chunk(self, chunk: TranscriptChunk, session: SessionView) -> ControllerDecision:
        if self._fail_mode == "raise":
            raise RuntimeError("MockController injected failure")
        if self._fail_mode == "timeout":
            await asyncio.sleep(3600)

        text = self._accumulated.get(chunk.utterance_id, "") + chunk.text
        self._accumulated[chunk.utterance_id] = text
        stripped = text.strip().lower()

        if stripped in _CHITCHAT:
            return ControllerDecision(Decision.SUPPRESS, TurnKind.CHIT_CHAT, "chit_chat", 1.0)

        if session.has_answer() and any(v in stripped for v in _PRESENTATION_VERBS):
            return ControllerDecision(Decision.SUPPRESS, TurnKind.PRESENTATION_ONLY,
                                       "presentation_restructure", 1.0)

        tokens = _tokenize(text)
        seen = self._seen_by_utterance.setdefault(chunk.utterance_id, set())

        # Word-boundary check: a naive substring test would treat "booking" as
        # matching "book" and wrongly disqualify a genuine refinement.
        _new_topic_re = re.compile(r"\b(?:i need|plan a|book)\b")
        is_refinement = session.has_answer() and not chunk.is_final and len(tokens) > 0 and \
            not _new_topic_re.search(stripped)

        if not chunk.is_final and len(tokens) < 3:
            return ControllerDecision(Decision.WAIT, TurnKind.NEW_REQUEST, "intent_unstable", 0.2)

        clauses = [c.strip() for c in re.split(r",| and ", text) if c.strip()]
        new_clauses = [c for c in clauses if _tokenize(c) - seen]

        if not new_clauses:
            if chunk.is_final:
                return ControllerDecision(Decision.WAIT, TurnKind.NEW_REQUEST, "no_new_content", 0.5)
            return ControllerDecision(Decision.WAIT, TurnKind.NEW_REQUEST, "intent_unstable", 0.4)

        turn_kind = TurnKind.REFINEMENT if is_refinement else TurnKind.NEW_REQUEST
        trigger = "refinement" if is_refinement else ("final" if chunk.is_final and len(clauses) <= 1 else
                                                        ("multi_intent" if len(new_clauses) > 1 else "provisional"))
        sub_queries = []
        for i, clause in enumerate(new_clauses):
            qid = f"{chunk.utterance_id}.q{len(seen) + i + 1}"
            sub_queries.append(SubQuery(
                query_id=qid, text=clause, intent_label=clause[:40],
                trigger=trigger, utterance_id=chunk.utterance_id,
            ))
            seen |= _tokenize(clause)

        stability = 0.9 if chunk.is_final else 0.65
        return ControllerDecision(
            decision=Decision.RETRIEVE, turn_kind=turn_kind,
            reason="multi_intent" if len(sub_queries) > 1 else "provisional_stable",
            stability=stability, sub_queries=tuple(sub_queries),
        )


class MockRetriever:
    """In-memory retriever over a tiny fixed corpus, keyword-overlap scored.
    Deterministic ordering (ties broken by chunk_id) so tests are stable.
    """

    def __init__(self, corpus: list[Chunk] | None = None, latency_ms: float = 50.0,
                 fail_mode: FailMode = "none", low_confidence_threshold: float = 0.15):
        self._corpus = corpus if corpus is not None else _default_corpus()
        self._by_id = {c.chunk_id: c for c in self._corpus}
        self._latency_ms = latency_ms
        self._fail_mode = fail_mode
        self._low_confidence_threshold = low_confidence_threshold

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    async def search(self, queries: list[SubQuery], k: int = 5) -> list[RetrievalResult]:
        if self._fail_mode == "raise":
            raise RuntimeError("MockRetriever injected failure")
        if self._fail_mode == "timeout":
            await asyncio.sleep(3600)

        await asyncio.sleep(self._latency_ms / 1000)

        if self._fail_mode == "empty":
            return [RetrievalResult(query_id=q.query_id, evidence=(), latency_ms=self._latency_ms,
                                     low_confidence=True) for q in queries]
        if self._fail_mode == "malformed":
            bogus = Chunk(chunk_id="Doc_999 §0 #0", doc_id="Doc_999", section="0", text="")
            return [RetrievalResult(query_id=q.query_id,
                                     evidence=(EvidenceChunk(chunk=bogus, score=1.0, query_ids=(q.query_id,)),),
                                     latency_ms=self._latency_ms, low_confidence=False) for q in queries]

        results = []
        for q in queries:
            q_tokens = _tokenize(q.text)
            scored = []
            for c in self._corpus:
                overlap = len(q_tokens & _tokenize(c.text))
                if overlap:
                    scored.append((overlap / max(len(q_tokens), 1), c))
            scored.sort(key=lambda t: (-t[0], t[1].chunk_id))
            top = scored[:k]
            best = top[0][0] if top else 0.0
            evidence = tuple(
                EvidenceChunk(chunk=c, score=score, query_ids=(q.query_id,), dense_rank=i, sparse_rank=i)
                for i, (score, c) in enumerate(top)
            )
            results.append(RetrievalResult(
                query_id=q.query_id, evidence=evidence, latency_ms=self._latency_ms,
                low_confidence=best < self._low_confidence_threshold,
            ))
        return results


class MockSynthesizer:
    """Deterministic synthesizer: concatenates retrieved snippets into a
    templated answer with citations, never inventing facts. Tracks version
    numbers per session in-process (ephemeral, no cross-session state).
    """

    def __init__(self, llm=None, telemetry=None, fail_mode: FailMode = "none"):
        self._llm = llm
        self._telemetry = telemetry
        self._fail_mode = fail_mode
        self._versions: dict[str, int] = {}
        self._history: dict[str, AnswerVersion] = {}

    def _next_version(self, session_id: str) -> int:
        self._versions[session_id] = self._versions.get(session_id, 0) + 1
        return self._versions[session_id]

    async def synthesize(self, session_id: str, utterance: str,
                          sub_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        return await self._build(session_id, sub_queries, results, parent=None, change_kind="initial")

    async def refine(self, session_id: str, utterance: str,
                      delta_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        prior = self._history.get(session_id)
        parent_claims = list(prior.claims) if prior else []
        parent_sub_queries = list(prior.sub_queries) if prior else []
        av = await self._build(session_id, delta_queries, results,
                                parent=prior, change_kind="refinement",
                                extra_claims=parent_claims, extra_sub_queries=parent_sub_queries)
        return av

    async def restructure(self, session_id: str, instruction: str) -> AnswerVersion:
        prior = self._history.get(session_id)
        if prior is None:
            raise RuntimeError("restructure called with no prior answer in session")
        version = self._next_version(session_id)
        text = " • " + " • ".join(c.text for c in prior.claims) if prior.claims else prior.text
        av = AnswerVersion(
            version=version, parent_version=prior.version, change_kind="restructure",
            text=text, claims=list(prior.claims), citations=list(prior.citations),
            evidence_ids=list(prior.evidence_ids), sub_queries=list(prior.sub_queries),
            uncertainty=prior.uncertainty, created_ms=int(time.monotonic() * 1000),
        )
        self._history[session_id] = av
        return av

    async def _build(self, session_id: str, sub_queries: list[SubQuery], results: list[RetrievalResult],
                      parent: AnswerVersion | None, change_kind: str,
                      extra_claims: list[Claim] | None = None,
                      extra_sub_queries: list[SubQuery] | None = None) -> AnswerVersion:
        if self._fail_mode == "raise":
            raise RuntimeError("MockSynthesizer injected failure")

        claims: list[Claim] = list(extra_claims or [])
        uncertain_parts = []
        for q, r in zip(sub_queries, results):
            if not r.evidence:
                uncertain_parts.append(q.intent_label)
                continue
            top = r.evidence[0]
            claims.append(Claim(text=f"{q.intent_label}: {top.chunk.text[:120]}",
                                 citations=[top.chunk.citation], supported=True))

        citations = sorted({c for claim in claims for c in claim.citations})
        evidence_ids = sorted({e.chunk.chunk_id for r in results for e in r.evidence})
        text = " ".join(c.text for c in claims) if claims else "I don't have enough information yet."
        uncertainty = None
        if uncertain_parts:
            uncertainty = f"Could not verify from the retrieved corpus: {', '.join(uncertain_parts)}."

        version = self._next_version(session_id)
        av = AnswerVersion(
            version=version, parent_version=parent.version if parent else None,
            change_kind=change_kind, text=text, claims=claims, citations=citations,
            evidence_ids=evidence_ids, sub_queries=list(extra_sub_queries or []) + list(sub_queries),
            uncertainty=uncertainty, created_ms=int(time.monotonic() * 1000),
        )
        self._history[session_id] = av
        return av


def _default_corpus() -> list[Chunk]:
    return [
        Chunk(chunk_id="Doc_12 §2 #1", doc_id="Doc_12", section="2",
              text="Venue A in Pune supports workshop capacity up to 40 attendees with tiered seating."),
        Chunk(chunk_id="Doc_31 §4 #1", doc_id="Doc_31", section="4",
              text="Cancellation policy: bookings may be cancelled up to 72 hours before the event for a full refund."),
        Chunk(chunk_id="Doc_09 §1 #1", doc_id="Doc_09", section="1",
              text="Catering options include on-site buffet service and external vendor catering with prior approval."),
        Chunk(chunk_id="Doc_20 §1 #1", doc_id="Doc_20", section="1",
              text="Standard employee travel reimbursement covers economy airfare, lodging and meals with receipts."),
        Chunk(chunk_id="Doc_20 §3 #1", doc_id="Doc_20", section="3",
              text="International travel requires senior director approval and mandatory foreign-currency receipt verification."),
        Chunk(chunk_id="Doc_20 §5 #1", doc_id="Doc_20", section="5",
              text="Late-booking exception: reimbursement for bookings made after travel requires senior director approval."),
    ]
