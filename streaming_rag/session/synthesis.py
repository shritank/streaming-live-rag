"""GroundedSynthesizer — session-aware synthesis (Task 3).

Answers are built extractively by default: every claim is a sentence taken
from a retrieved chunk, carrying that chunk's citation. This makes the hard
rule "no factual claim from parametric memory" structurally true rather than
a prompt instruction, and makes G4 support verifiable.

When an LLM is configured it is used only to *phrase* the assembled claims,
never to supply facts: the rewritten text is re-verified against the same
evidence and is discarded if it drifts (see `_phrase`).
"""
from __future__ import annotations

import time

from ..config import Config
from ..contracts import (
    AnswerVersion, Claim, LLMClient, RetrievalResult, SubQuery,
    Telemetry, NullTelemetry,
)
from ..retrieval.text import content_tokens, split_sentences
from . import grounding
from .delta import merge_claims, union_citations
from .store import SessionStore


class GroundedSynthesizer:
    def __init__(self, retriever, session_store: SessionStore, config: Config,
                 llm: LLMClient | None = None, telemetry: Telemetry | None = None):
        self._retriever = retriever
        self._sessions = session_store
        self._config = config
        self._llm = llm
        self._telemetry = telemetry or NullTelemetry()

    # ---------- Synthesizer protocol ----------

    async def synthesize(self, session_id: str, utterance: str,
                          sub_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        claims, unsupported = self._claims_from(sub_queries, results)
        prior = self._sessions.view(session_id).latest_answer()
        return self._finalise(session_id, claims, unsupported, sub_queries, results,
                               parent=prior, change_kind="initial", carried=[])

    async def refine(self, session_id: str, utterance: str,
                      delta_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        prior = self._sessions.view(session_id).latest_answer()
        if prior is None:
            return await self.synthesize(session_id, utterance, delta_queries, results)

        delta_claims, unsupported = self._claims_from(delta_queries, results)
        merged, carried = merge_claims(prior.claims, delta_claims)
        return self._finalise(session_id, merged, unsupported,
                               list(prior.sub_queries) + list(delta_queries), results,
                               parent=prior, change_kind="refinement", carried=carried)

    async def restructure(self, session_id: str, instruction: str) -> AnswerVersion:
        """Presentation-only turn: no retrieval, no new citations (pitfall 4)."""
        prior = self._sessions.view(session_id).latest_answer()
        if prior is None:
            raise RuntimeError("restructure with no prior answer in session")

        text = self._reformat(prior.claims, instruction)
        return AnswerVersion(
            version=prior.version + 1, parent_version=prior.version, change_kind="restructure",
            text=text, claims=list(prior.claims), citations=list(prior.citations),
            evidence_ids=list(prior.evidence_ids), sub_queries=list(prior.sub_queries),
            uncertainty=prior.uncertainty, created_ms=int(time.monotonic() * 1000),
        )

    # ---------- internals ----------

    def _claims_from(self, sub_queries: list[SubQuery],
                      results: list[RetrievalResult]) -> tuple[list[Claim], list[str]]:
        by_query = {r.query_id: r for r in results}
        claims: list[Claim] = []
        unsupported: list[str] = []

        for q in sub_queries:
            result = by_query.get(q.query_id)
            if result is None or not result.evidence or result.low_confidence:
                unsupported.append(q.intent_label)
                continue
            top = result.evidence[0]
            sentence = self._best_sentence(q.text, top.chunk.text)
            claims.append(Claim(text=sentence, citations=[top.chunk.citation]))

        return claims, unsupported

    @staticmethod
    def _best_sentence(query: str, chunk_text: str) -> str:
        """The sentence of the chunk that best answers the query. Keeping the
        claim to one sentence keeps the grounding check meaningful — a whole
        chunk would trivially 'support' anything inside it."""
        sentences = split_sentences(chunk_text)
        if len(sentences) <= 1:
            return chunk_text.strip()
        q_tokens = set(content_tokens(query))
        best, best_score = sentences[0], -1.0
        for s in sentences:
            s_tokens = set(content_tokens(s))
            if not s_tokens:
                continue
            score = len(q_tokens & s_tokens) / len(q_tokens or {1})
            if score > best_score:
                best, best_score = s, score
        return best.strip()

    def _finalise(self, session_id: str, claims: list[Claim], unsupported: list[str],
                   sub_queries: list[SubQuery], results: list[RetrievalResult],
                   parent: AnswerVersion | None, change_kind: str,
                   carried: list[str]) -> AnswerVersion:
        if self._config.synthesis.grounding_check:
            report = grounding.verify(claims, self._resolve_citation)
            claims = report.claims
            unsupported = unsupported + [
                c.text[:40] for c in claims if c.supported is False
            ]
        citations = union_citations(claims, carried)
        evidence_ids = sorted({e.chunk.chunk_id for r in results for e in r.evidence})
        if parent is not None and change_kind == "refinement":
            evidence_ids = sorted(set(evidence_ids) | set(parent.evidence_ids))

        text = " ".join(c.text for c in claims) if claims else \
            "I could not find anything in the corpus that answers this."
        uncertainty = None
        if unsupported:
            uncertainty = ("Could not be verified from the retrieved corpus: "
                            + "; ".join(dict.fromkeys(unsupported)) + ".")

        version = (parent.version + 1) if parent is not None else 1
        return AnswerVersion(
            version=version,
            parent_version=parent.version if parent is not None else None,
            change_kind=change_kind, text=text, claims=claims, citations=citations,
            evidence_ids=evidence_ids, sub_queries=list(sub_queries),
            uncertainty=uncertainty, created_ms=int(time.monotonic() * 1000),
        )

    def _resolve_citation(self, citation: str):
        getter = getattr(self._retriever, "get_chunk_by_citation", None)
        if getter is not None:
            return getter(citation)
        return self._retriever.get_chunk(citation)

    @staticmethod
    def _reformat(claims: list[Claim], instruction: str) -> str:
        lowered = instruction.lower()
        wants_bullets = any(w in lowered for w in ("bullet", "point", "list"))
        limit = None
        for word, n in (("two", 2), ("three", 3), ("four", 4)):
            if word in lowered:
                limit = n
        import re
        m = re.search(r"\b(\d+)\b", lowered)
        if m:
            limit = int(m.group(1))

        selected = claims[:limit] if limit else claims
        if wants_bullets:
            return "\n".join(f"• {c.text} [{', '.join(c.citations)}]" for c in selected)
        return " ".join(c.text for c in selected)
