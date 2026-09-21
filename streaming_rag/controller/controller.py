"""RetrievalController — the WAIT / RETRIEVE / SUPPRESS policy (Task 2).

Decision order per chunk:

  1. classify the turn (new request / refinement / presentation-only / chit-chat)
  2. presentation-only and chit-chat  -> SUPPRESS, retrieval_required = false
  3. score intent stability           -> below threshold, WAIT
  4. decompose into sub-queries, drop anything already issued this utterance
  5. nothing new left                 -> WAIT, else RETRIEVE

`controller.mode` selects how step 1 and step 4 are done:
  rule   - structural heuristics only (default: fast, deterministic, no tokens)
  model  - the LLM classifies and decomposes
  hybrid - rules decide, and the LLM is consulted only when the rule path is
           uncertain (stability sits near the threshold)
"""
from __future__ import annotations

import json
import re

from ..config import Config
from ..contracts import (
    ControllerDecision, Decision, LLMClient, SessionView, SubQuery,
    Telemetry, NullTelemetry, TranscriptChunk, TurnKind,
)
from ..retrieval.text import content_tokens, jaccard
from . import stability
from .decompose import decompose

_PRESENTATION_RE = re.compile(
    r"\b(repeat|rephrase|reword|summari[sz]e|shorten|condense|expand|translate|"
    r"say (?:that|it) again|in (?:two|three|\d+) (?:bullets?|points?|lines?|sentences?)|"
    r"as (?:a )?(?:bullet|list)|bullet points?|tl;?dr)\b",
    re.IGNORECASE,
)
_CHITCHAT_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|thankyou|cheers|ok|okay|got it|great|"
    r"perfect|bye|goodbye|good morning|good afternoon)[\s!.,]*$",
    re.IGNORECASE,
)
_REFINEMENT_RE = re.compile(
    r"\b(actually|also|wait|but |instead|what if|in addition|one more thing|"
    r"it was|they were|turns out|forgot to (?:say|mention)|make it)\b",
    re.IGNORECASE,
)

_SYSTEM = (
    "You classify a live transcript fragment for a retrieval system. "
    "Reply with JSON only."
)


class RetrievalController:
    def __init__(self, llm: LLMClient | None, telemetry: Telemetry, config: Config | dict):
        self._llm = llm
        self._telemetry = telemetry or NullTelemetry()
        self._config = config if isinstance(config, Config) else Config()
        self._accumulated: dict[str, str] = {}
        self._previous: dict[str, str] = {}
        self._covered: dict[str, set[str]] = {}
        self._issued: dict[str, int] = {}

    def reset_utterance(self, utterance_id: str) -> None:
        for d in (self._accumulated, self._previous, self._covered, self._issued):
            d.pop(utterance_id, None)

    async def on_chunk(self, chunk: TranscriptChunk, session: SessionView) -> ControllerDecision:
        uid = chunk.utterance_id
        previous = self._accumulated.get(uid, "")
        text = previous + chunk.text
        self._accumulated[uid] = text
        self._previous[uid] = previous
        covered = self._covered.setdefault(uid, set())

        utterance = text.strip()
        if not utterance and not chunk.is_final:
            return ControllerDecision(Decision.WAIT, TurnKind.NEW_REQUEST, "empty_fragment", 0.0)

        turn_kind = self._classify(utterance, session)

        if turn_kind == TurnKind.CHIT_CHAT:
            return ControllerDecision(Decision.SUPPRESS, TurnKind.CHIT_CHAT, "chit_chat", 1.0)

        if turn_kind == TurnKind.PRESENTATION_ONLY:
            return ControllerDecision(Decision.SUPPRESS, TurnKind.PRESENTATION_ONLY,
                                       "presentation_restructure", 1.0)

        confidence = stability.score(utterance, previous, chunk.is_final)
        threshold = (self._config.controller.min_stability if chunk.is_final
                      else self._config.controller.provisional_stability)
        if confidence < threshold:
            return ControllerDecision(Decision.WAIT, turn_kind, "intent_unstable", confidence)

        trigger = self._trigger_for(turn_kind, chunk.is_final)
        candidates = decompose(utterance, covered, uid, trigger, self._issued.get(uid, 0))

        if turn_kind == TurnKind.REFINEMENT:
            candidates = self._scope_to_delta(candidates, session, covered)

        if not candidates:
            return ControllerDecision(Decision.WAIT, turn_kind, "no_new_intent", confidence)

        if len(candidates) > 1:
            trigger = "refinement" if turn_kind == TurnKind.REFINEMENT else "multi_intent"
            candidates = [_retrigger(q, trigger) for q in candidates]

        for q in candidates:
            covered |= set(content_tokens(q.intent_label)) | set(content_tokens(q.text))
        self._issued[uid] = self._issued.get(uid, 0) + len(candidates)

        reason = ("refinement_delta" if turn_kind == TurnKind.REFINEMENT
                   else "multi_intent" if len(candidates) > 1
                   else "intent_stable")
        return ControllerDecision(
            decision=Decision.RETRIEVE, turn_kind=turn_kind, reason=reason,
            stability=confidence, sub_queries=tuple(candidates),
        )

    def _trigger_for(self, turn_kind: TurnKind, is_final: bool) -> str:
        if turn_kind == TurnKind.REFINEMENT:
            return "refinement"
        return "final" if is_final else "provisional"

    def _classify(self, utterance: str, session: SessionView) -> TurnKind:
        if _CHITCHAT_RE.match(utterance):
            return TurnKind.CHIT_CHAT
        has_prior = session.has_answer()
        if has_prior and _PRESENTATION_RE.search(utterance):
            return TurnKind.PRESENTATION_ONLY
        if has_prior and self._looks_like_refinement(utterance, session):
            return TurnKind.REFINEMENT
        return TurnKind.NEW_REQUEST

    def _looks_like_refinement(self, utterance: str, session: SessionView) -> bool:
        if _REFINEMENT_RE.search(utterance):
            return True
        # A short follow-up that shares vocabulary with the open topic is a
        # constraint on it, not a new request.
        tokens = set(content_tokens(utterance))
        if len(tokens) > 12:
            return False
        topic = set(content_tokens(session.topic_summary()))
        return bool(topic) and jaccard(tokens, topic) >= 0.12

    def _scope_to_delta(self, candidates: list[SubQuery], session: SessionView,
                         covered: set[str]) -> list[SubQuery]:
        """A refinement must query only what the new constraint changes, never
        re-run the whole topic (G5). Anything already answered is dropped, and
        what survives is linked to the prior query it refines."""
        prior = session.prior_sub_queries()
        prior_tokens = [set(content_tokens(q.text)) for q in prior]
        out = []
        for q in candidates:
            tokens = set(content_tokens(q.text))
            parent = None
            best = 0.0
            for pq, ptok in zip(prior, prior_tokens):
                overlap = jaccard(tokens, ptok)
                if overlap > best:
                    best, parent = overlap, pq
            if best >= 0.75:
                continue                       # identical to an answered query
            out.append(SubQuery(
                query_id=q.query_id, text=q.text, intent_label=q.intent_label,
                trigger="refinement", utterance_id=q.utterance_id,
                parent_query_id=parent.query_id if parent else None,
            ))
        return out


def _retrigger(q: SubQuery, trigger: str) -> SubQuery:
    return SubQuery(query_id=q.query_id, text=q.text, intent_label=q.intent_label,
                     trigger=trigger, utterance_id=q.utterance_id,
                     parent_query_id=q.parent_query_id)
