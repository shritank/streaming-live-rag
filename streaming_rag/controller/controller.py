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
    def __init__(self, llm: LLMClient | None, telemetry: Telemetry, config: Config | dict,
                 corpus_vocab=None, corpus_bigrams=None):
        """`corpus_vocab` and `corpus_bigrams` are optional zero-arg callables
        returning the corpus's discriminative-term vocabulary and low-DF
        bigrams (see HybridRetriever.specific_vocabulary /
        HybridRetriever.low_df_bigrams), used only for the weak-anchor
        heuristic in stability scoring. Kept as injected callables rather
        than a Retriever reference so the controller stays decoupled from the
        Retriever protocol (§6.2 of the project contracts)."""
        self._llm = llm
        self._telemetry = telemetry or NullTelemetry()
        self._config = config if isinstance(config, Config) else Config()
        self._corpus_vocab = corpus_vocab
        self._corpus_bigrams = corpus_bigrams
        self._accumulated: dict[str, str] = {}
        self._previous: dict[str, str] = {}
        self._covered: dict[str, set[str]] = {}
        self._issued: dict[str, int] = {}
        # the sub-queries issued by the MOST RECENT retrieve decision for each
        # utterance, so a growing clause's next (more complete) sub-query can
        # be linked via parent_query_id to the stale partial one it supersedes
        self._last_issued: dict[str, list[SubQuery]] = {}

    def reset_utterance(self, utterance_id: str) -> None:
        for d in (self._accumulated, self._previous, self._covered, self._issued, self._last_issued):
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

        known_terms = self._corpus_vocab() if self._corpus_vocab is not None else None
        low_df_bigrams = self._corpus_bigrams() if self._corpus_bigrams is not None else None
        confidence = stability.score(utterance, previous, chunk.is_final, known_terms, low_df_bigrams)
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

        # Within-utterance supersession: as a single clause grows across
        # chunks ("Before which..." -> "...did Chagatai publicly" -> "...
        # dispute Jochi's paternity?"), decompose() naturally re-emits it each
        # time new words arrive. Without this, every partial version stays a
        # live, separate sub-query — the engine never cancels the earlier
        # (incomplete, sometimes wrong-topic) retrieval, and synthesis ends up
        # citing stale evidence alongside the final, correct one. Link each
        # new candidate to the most recent sub-query it overlaps heavily with
        # from THIS utterance, so the engine cancels the stale one (§ engine.
        # _dispatch_decision's parent_query_id handling).
        candidates = self._link_supersession(uid, candidates)
        self._last_issued[uid] = list(candidates)

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

    def _link_supersession(self, uid: str, candidates: list[SubQuery],
                            threshold: float = 0.3) -> list[SubQuery]:
        """Within-utterance supersession must take priority over any
        cross-turn `parent_query_id` `_scope_to_delta` already set: they mean
        different things (a same-utterance stale retrieval to cancel, vs. a
        cross-turn refinement lineage marker for citation bookkeeping), but
        share the same single field. `engine.py`'s cancellation check only
        ever looks the id up in the CURRENT utterance's own pending-task
        dict, so a cross-turn id is always a harmless no-op there — silently
        skipping this link whenever `_scope_to_delta` had already set one
        (the previous behaviour) left a same-utterance stale retrieval
        uncancelled and its evidence reached the synthesizer regardless.
        Overriding is therefore always safe: it can only enable a
        cancellation engine.py would otherwise have silently ignored."""
        prior = self._last_issued.get(uid, [])
        if not prior:
            return candidates
        prior_tokens = [set(content_tokens(q.text)) for q in prior]
        linked = []
        for q in candidates:
            tokens = set(content_tokens(q.text))
            best_idx, best = None, 0.0
            for i, ptok in enumerate(prior_tokens):
                overlap = jaccard(tokens, ptok)
                if overlap > best:
                    best, best_idx = overlap, i
            if best_idx is not None and best >= threshold:
                linked.append(SubQuery(query_id=q.query_id, text=q.text, intent_label=q.intent_label,
                                        trigger=q.trigger, utterance_id=q.utterance_id,
                                        parent_query_id=prior[best_idx].query_id))
            else:
                linked.append(q)
        return linked

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
