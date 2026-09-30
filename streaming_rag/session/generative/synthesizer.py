"""Session-aware synthesis (pipeline component [4]).

Three modes, chosen by the controller's turn classification:

* ``synthesize``  - a new request: write a fresh, cited answer.
* ``refine``      - a late constraint: update the existing answer in place and
                    issue version N+1, keeping the claims it did not affect.
* ``restructure`` - a presentation-only turn: reformat what is already there,
                    with no retrieval and no new facts.

Claims are generated first and prose second, because a list of claims can be
verified one at a time while a paragraph cannot. Nothing unverified is ever
streamed as fact.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ...contracts import (
    AnswerVersion,
    Claim,
    EvidenceChunk,
    LLMClient,
    NullTelemetry,
    RetrievalResult,
    SubQuery,
    Telemetry,
)
from . import prompts
from .delta import DeltaEngine
from .grounding import GroundingVerifier, _first_json_blob
from .settings import get, load_config
from .store import SessionStore
from .uncertainty import build_uncertainty

# the engine passes a plain function; tests pass a coroutine
OnPartial = Callable[[str], Awaitable[None] | None]


class GroundedSynthesizer:
    """Implements contracts.Synthesizer.

    ``emits_answer_version`` tells the engine not to emit
    ``answer_version_created`` itself: project context §6.5 assigns that event
    to Task 3, and two copies per turn would corrupt the G6 trace.

    Deliberately holds **no retriever**: this component cannot search even if it
    wanted to, which is what makes "restructure performs zero retrieval" and
    "refine only uses delta evidence" structural rather than a promise.
    """

    emits_answer_version = True

    def __init__(self, store: SessionStore, llm: LLMClient,
                 verifier: GroundingVerifier | None = None,
                 telemetry: Telemetry | None = None,
                 config: dict[str, Any] | None = None,
                 on_partial: OnPartial | None = None) -> None:
        self.cfg = config or load_config()
        self.store = store
        self.llm = llm
        self.telemetry = telemetry or NullTelemetry()
        self.verifier = verifier or GroundingVerifier(config=self.cfg,
                                                      telemetry=self.telemetry, llm=llm)
        self.on_partial = on_partial
        self.delta_engine = DeltaEngine(llm, self.cfg)
        self._max_chars = int(get(self.cfg, "synthesis.max_chunk_chars", 900))
        self._max_claims = int(get(self.cfg, "synthesis.max_claims", 8))
        self._max_evidence = int(get(self.cfg, "synthesis.max_evidence_chunks", 12))

    # ------------------------------------------------------------------
    # mode A: a new request
    # ------------------------------------------------------------------
    async def synthesize(self, session_id: str, utterance: str,
                         sub_queries: list[SubQuery],
                         results: list[RetrievalResult],
                         on_partial: OnPartial | None = None) -> AnswerVersion:
        self._begin_turn(session_id, utterance, sub_queries, results)
        state = self.store.state(session_id)
        evidence = self._rank_evidence(self.store.evidence_for(session_id))

        raw_claims, claim_queries = await self._generate_claims(
            system=prompts.SYNTHESIS_SYSTEM,
            prompt=prompts.build_synthesis_prompt(
                utterance, sub_queries, evidence, self._max_chars, self._max_claims),
            purpose="synthesis",
        )
        kept, report = await self.verifier.verify_with_report(
            raw_claims, self.store.evidence_for(session_id))

        outcome = build_uncertainty(
            sub_queries, kept, claim_queries,
            set(state.low_confidence_query_ids), report.dropped_claims)

        if outcome.needs_clarification:
            return await self._clarification_version(
                session_id, utterance, sub_queries, results, outcome.unsupported_intents,
                on_partial)

        text = await self._render(kept, outcome.text, on_partial)
        return self._commit(
            session_id, change_kind="initial", text=text, claims=kept,
            sub_queries=sub_queries, uncertainty=outcome.text,
            results=results, report=report, unsupported=outcome.unsupported_intents)

    # ------------------------------------------------------------------
    # mode B: a late constraint
    # ------------------------------------------------------------------
    async def refine(self, session_id: str, utterance: str,
                     delta_queries: list[SubQuery],
                     results: list[RetrievalResult],
                     on_partial: OnPartial | None = None) -> AnswerVersion:
        previous = self.store.latest_version(session_id)
        if previous is None:
            # Nothing to refine: treat it as a new request rather than failing.
            return await self.synthesize(session_id, utterance, delta_queries,
                                         results, on_partial)

        self._begin_turn(session_id, utterance, delta_queries, results)
        state = self.store.state(session_id)
        # Only the delta evidence goes into the decision prompt; everything else
        # is already represented by the previous claims.
        delta_evidence = self._rank_evidence(self._evidence_from(results))

        plan = await self.delta_engine.plan(
            previous, utterance, delta_queries, delta_evidence, self._max_chars)
        new_claims, new_claim_queries = await self._generate_claims(
            system=prompts.SYNTHESIS_SYSTEM,
            prompt=prompts.build_new_claims_prompt(
                utterance, delta_queries, delta_evidence, self._max_chars),
            purpose="refinement_claims",
        )

        # Surviving claims keep their original citations, which is what "prior
        # citations are preserved" means in G5; only the new ones need checking
        # against the evidence, but re-verifying everything is cheap insurance.
        candidates = plan.surviving_claims() + new_claims
        kept, report = await self.verifier.verify_with_report(
            candidates, self.store.evidence_for(session_id))

        offset = len(plan.surviving_claims())
        claim_queries = {i + offset: q for i, q in new_claim_queries.items()}
        # Claims carried over from V1 have no sub_query_id of their own, so their
        # coverage is inferred from which sub-queries retrieved the chunks they
        # cite. Without this every refinement would hedge about intents the
        # surviving claims still answer.
        claim_queries.update(self._infer_claim_queries(session_id, kept, skip=claim_queries))
        all_queries = self.store.view(session_id).prior_sub_queries()
        outcome = build_uncertainty(
            all_queries, kept, claim_queries,
            set(state.low_confidence_query_ids), report.dropped_claims)

        text = await self._render(kept, outcome.text, on_partial)
        version = self._commit(
            session_id, change_kind="refinement", text=text, claims=kept,
            sub_queries=delta_queries, uncertainty=outcome.text, results=results,
            report=report, unsupported=outcome.unsupported_intents,
            extra={"delta_counts": plan.counts(), "delta_fallback": plan.used_fallback})
        return version

    # ------------------------------------------------------------------
    # mode C: presentation only
    # ------------------------------------------------------------------
    async def restructure(self, session_id: str, instruction: str,
                          on_partial: OnPartial | None = None) -> AnswerVersion:
        previous = self.store.latest_version(session_id)
        if previous is None:
            return self._commit(
                session_id, change_kind="clarification",
                text="I do not have a previous answer to reformat yet. "
                     "What would you like me to look up?",
                claims=[], sub_queries=[], uncertainty=None, results=[],
                report=None, unsupported=[])

        self.store.add_turn(session_id, instruction)
        text = previous.text
        try:
            response = await self.llm.complete(
                purpose="restructure", system=prompts.RESTRUCTURE_SYSTEM,
                prompt=prompts.build_restructure_prompt(previous, instruction),
                json_mode=True, max_tokens=800,
            )
            blob = _first_json_blob(response.text)
            if blob:
                payload = json.loads(blob)
                candidate = str(payload.get("text", "")).strip()
                if candidate:
                    text = candidate
        except Exception as exc:  # noqa: BLE001 - keep the old text on failure
            self.telemetry.emit("error", where="synthesizer.restructure",
                                error_type=type(exc).__name__, message=str(exc)[:200],
                                session_id=session_id)

        # Claims and citations are carried over untouched: a reformat may not
        # introduce a fact, so it may not introduce a source either.
        claims = [Claim(text=c.text, citations=list(c.citations), supported=c.supported)
                  for c in previous.claims]
        await self._emit_partial(text, on_partial)
        return self._commit(
            session_id, change_kind="restructure", text=text, claims=claims,
            sub_queries=previous.sub_queries, uncertainty=previous.uncertainty,
            results=[], report=None, unsupported=[],
            evidence_ids=list(previous.evidence_ids))

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _begin_turn(self, session_id: str, utterance: str,
                    sub_queries: list[SubQuery], results: list[RetrievalResult]) -> None:
        if not self.store.is_open(session_id):
            self.store.open(session_id)
        self.store.add_turn(session_id, utterance)
        self.store.add_sub_queries(session_id, sub_queries)
        self.store.add_evidence(session_id, results)

    @staticmethod
    def _evidence_from(results: list[RetrievalResult]) -> dict[str, EvidenceChunk]:
        evidence: dict[str, EvidenceChunk] = {}
        for result in results:
            for ev in result.evidence:
                current = evidence.get(ev.chunk.chunk_id)
                if current is None or ev.score > current.score:
                    evidence[ev.chunk.chunk_id] = ev
        return evidence

    def _infer_claim_queries(self, session_id: str, claims: list[Claim],
                             skip: dict[int, str | None]) -> dict[int, str | None]:
        """Map claims to sub-queries via the citations they carry."""
        by_citation: dict[str, str] = {}
        for ev in self.store.evidence_for(session_id).values():
            if ev.query_ids:
                by_citation.setdefault(ev.chunk.citation, ev.query_ids[0])
        inferred: dict[int, str | None] = {}
        for index, claim in enumerate(claims):
            if skip.get(index):
                continue
            for citation in claim.citations:
                if citation in by_citation:
                    inferred[index] = by_citation[citation]
                    break
        return inferred

    def _rank_evidence(self, evidence: dict[str, EvidenceChunk]) -> list[EvidenceChunk]:
        ranked = sorted(evidence.values(), key=lambda ev: ev.score, reverse=True)
        return ranked[:self._max_evidence]

    async def _generate_claims(self, *, system: str, prompt: str,
                               purpose: str) -> tuple[list[Claim], dict[int, str | None]]:
        """Ask for claims as JSON and parse defensively."""
        try:
            response = await self.llm.complete(
                purpose=purpose, system=system, prompt=prompt,
                json_mode=True, max_tokens=1200,
            )
        except Exception as exc:  # noqa: BLE001 - a failed call yields no claims
            self.telemetry.emit("error", where=f"synthesizer.{purpose}",
                                error_type=type(exc).__name__, message=str(exc)[:200])
            return [], {}
        return self._parse_claims(response.text)

    def _parse_claims(self, raw: str) -> tuple[list[Claim], dict[int, str | None]]:
        blob = _first_json_blob(raw)
        if not blob:
            return [], {}
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            return [], {}
        rows = payload.get("claims") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return [], {}

        claims: list[Claim] = []
        queries: dict[int, str | None] = {}
        for row in rows[:self._max_claims]:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            raw_citations = row.get("citations")
            citations = [str(c).strip() for c in raw_citations
                         if str(c).strip()] if isinstance(raw_citations, list) else []
            queries[len(claims)] = (str(row["sub_query_id"])
                                    if row.get("sub_query_id") else None)
            claims.append(Claim(text=text, citations=citations))
        return claims, queries

    async def _render(self, claims: list[Claim], uncertainty: str | None,
                      on_partial: OnPartial | None = None) -> str:
        """Turn verified claims into prose with inline citation markers."""
        marker = str(get(self.cfg, "synthesis.citation_format", "[{citation}]"))
        sentences = []
        for claim in claims:
            text = claim.text.rstrip()
            if text and text[-1] not in ".!?":
                text += "."
            tags = " ".join(marker.format(citation=c) for c in claim.citations)
            sentences.append(f"{text} {tags}".strip())

        body = " ".join(sentences)
        if not body:
            body = ("I could not find support for this in the corpus."
                    if uncertainty else "No answer could be produced.")
        if uncertainty:
            body = f"{body}\n\nNot verified: {uncertainty}"
        await self._emit_partial(body, on_partial)
        return body

    async def _emit_partial(self, text: str, on_partial: OnPartial | None = None) -> None:
        """Stream the answer once it is verified (never before).

        The callback may be sync or async: the engine hands us a plain function
        that does `queue.put_nowait`, while tests use a coroutine.
        """
        callback = on_partial or self.on_partial
        if not callback or not get(self.cfg, "synthesis.stream_partials", True):
            return
        for sentence in _sentences(text):
            outcome = callback(sentence)
            if inspect.isawaitable(outcome):
                await outcome

    async def _clarification_version(self, session_id: str, utterance: str,
                                     sub_queries: list[SubQuery],
                                     results: list[RetrievalResult],
                                     unsupported: list[str],
                                     on_partial: OnPartial | None = None) -> AnswerVersion:
        question = ("I could not find anything about that in the corpus. "
                    "Could you give me more detail to search on?")
        try:
            response = await self.llm.complete(
                purpose="clarification", system=prompts.CLARIFY_SYSTEM,
                prompt=prompts.build_clarify_prompt(utterance, sub_queries),
                json_mode=True, max_tokens=120,
            )
            blob = _first_json_blob(response.text)
            if blob:
                candidate = str(json.loads(blob).get("question", "")).strip()
                if candidate:
                    question = candidate
        except Exception:  # noqa: BLE001 - the default question is fine
            pass
        await self._emit_partial(question, on_partial)
        return self._commit(
            session_id, change_kind="clarification", text=question, claims=[],
            sub_queries=sub_queries, uncertainty=None, results=results,
            report=None, unsupported=unsupported)

    def _commit(self, session_id: str, *, change_kind: str, text: str,
                claims: list[Claim], sub_queries: list[SubQuery],
                uncertainty: str | None, results: list[RetrievalResult],
                report: Any, unsupported: list[str],
                evidence_ids: list[str] | None = None,
                extra: dict[str, Any] | None = None) -> AnswerVersion:
        previous = self.store.latest_version(session_id)
        citations: list[str] = []
        for claim in claims:
            for citation in claim.citations:
                if citation not in citations:
                    citations.append(citation)

        if evidence_ids is None:
            cache = self.store.evidence_for(session_id)
            evidence_ids = [cid for cid, ev in cache.items()
                            if ev.chunk.citation in citations]

        version = AnswerVersion(
            version=self.store.next_version_number(session_id),
            parent_version=previous.version if previous else None,
            change_kind=change_kind,  # type: ignore[arg-type]
            text=text,
            claims=claims,
            citations=citations,
            evidence_ids=evidence_ids,
            sub_queries=list(sub_queries),
            uncertainty=uncertainty,
            created_ms=int(time.time() * 1000),
        )
        self.store.add_version(session_id, version)

        fields: dict[str, Any] = {
            "session_id": session_id, "version": version.version,
            "parent_version": version.parent_version, "change_kind": change_kind,
            "citations": citations, "n_claims": len(claims),
            "retriever_calls_for_version": len(results),
        }
        if extra:
            fields.update(extra)
        self.telemetry.emit("answer_version_created", **fields)

        if report is not None:
            self.telemetry.emit(
                "grounding_checked", session_id=session_id, version=version.version,
                n_claims=report.n_claims, n_supported=report.n_supported,
                fabricated_citations=report.fabricated_citations,
            )
        if uncertainty:
            self.telemetry.emit("uncertainty_flagged", session_id=session_id,
                                version=version.version, unsupported_intents=unsupported)
        return version


def _sentences(text: str) -> list[str]:
    """Split for streaming. Crude on purpose: the engine re-joins the pieces."""
    parts: list[str] = []
    buffer = ""
    for char in text:
        buffer += char
        if char in ".!?\n" and len(buffer.strip()) > 1:
            parts.append(buffer)
            buffer = ""
    if buffer:
        parts.append(buffer)
    return parts
