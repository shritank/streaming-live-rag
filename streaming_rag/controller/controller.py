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

import copy
import json
import re

from ..config import Config
from ..contracts import (
    ControllerDecision, Decision, LLMClient, SessionView, SubQuery,
    Telemetry, NullTelemetry, TranscriptChunk, TurnKind,
)
from ..retrieval.text import STOPWORDS, content_tokens, jaccard
from . import stability
from .decompose import decompose

_WORD_RE = re.compile(r"[a-z0-9']+")

# Presentation-only ("reformat what you already told me"): a formatting cue,
# NO topic words left once cue / filler words are removed, and either a
# reference back to the earlier answer or a leading command verb. A cue word
# alone is not enough - "Summarize the travel reimbursement rule" is a new
# request and "What is a list?" is a question.
_FORMAT_CUES = frozenset("""
    repeat rephrase reword paraphrase summarize summarise summary shorten shorter short brief briefly
    condense expand translate simplify simpler simply bullet bullets point points list table sentence
    sentences line lines word words again tldr format reformat differently concise concisely wording
""".split())
_LEADING_VERBS = frozenset("""
    repeat rephrase reword paraphrase summarize summarise shorten condense expand translate simplify
    reformat format tldr
""".split())
_ANAPHORA = frozenset("that it this these those them above previous last earlier answer response reply said".split())
_LANGUAGES = frozenset("""
    english hindi spanish french german tamil telugu kannada malayalam marathi bengali gujarati punjabi urdu
    korean japanese chinese mandarin arabic portuguese italian russian
""".split())
_PRESENTATION_FILLER = frozenset("""
    please kindly can could would will you your me my give make put say tell show write now just into way
    actually also and then ok okay well so one two three four five six seven eight nine ten
    text version form style manner
""".split()) | _LANGUAGES
_SKIPPED_LEADERS = frozenset("please kindly can could would will you now actually also and then ok okay well so just".split())
_QUANTIFIED_FORMAT_RE = re.compile(
    r"\bin (?:an? |one |two |three |four |five |\d+ )?(?:short |brief |few )?"
    r"(?:bullets?|points?|lines?|sentences?|words?)\b", re.IGNORECASE)

# Chit-chat: only greeting / thanks / closing words, at least one of them real.
_CHITCHAT_STRONG = frozenset(
    "hi hello hey thanks thank thankyou cheers bye goodbye appreciate appreciated morning afternoon evening night".split())
_CHITCHAT_WEAK = frozenset("ok okay great perfect cool awesome nice got".split())
_CHITCHAT_WORDS = _CHITCHAT_STRONG | _CHITCHAT_WEAK | frozenset("""
    got it good morning afternoon evening night alright sure that's thats all for today now so very much
    a lot you i'll see later talk soon take care helpful
""".split())


def is_presentation_request(utterance: str) -> bool:
    """True when the utterance only asks to reformat / repeat / shorten the
    previous answer (no new information need)."""
    tokens = _WORD_RE.findall(utterance.lower().replace("tl;dr", "tldr"))
    if not any(t in _FORMAT_CUES for t in tokens):
        return False
    residue = [t for t in tokens if t not in STOPWORDS and t not in _FORMAT_CUES and t not in _ANAPHORA
               and t not in _PRESENTATION_FILLER and not t.isdigit() and len(t) > 1]
    if residue:
        return False                      # topic words present: a real question, retrieve
    leader = next((t for t in tokens if t not in _SKIPPED_LEADERS), "")
    return (any(t in _ANAPHORA for t in tokens) or leader in _LEADING_VERBS
            or bool(_QUANTIFIED_FORMAT_RE.search(utterance)))


def is_chit_chat(utterance: str) -> bool:
    tokens = _WORD_RE.findall(utterance.lower())
    return (bool(tokens) and all(t in _CHITCHAT_WORDS for t in tokens)
            and any(t in _CHITCHAT_STRONG or t in _CHITCHAT_WEAK for t in tokens))

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
        # All per-utterance state is keyed by (session_id, utterance_id): two sessions may use the same
        # utterance id at the same time ("u1" is the guide's own example id) and must never share state.
        # Query ids stay "<utterance_id>.q<n>" (unique within a session, as the contract says).
        self._accumulated: dict[tuple[str, str], str] = {}
        self._previous: dict[tuple[str, str], str] = {}
        self._covered: dict[tuple[str, str], set[str]] = {}
        self._issued: dict[tuple[str, str], int] = {}
        # every still-live (not yet superseded) sub-query of each utterance,
        # and the clause each was built from, so a growing clause's next
        # (more complete) sub-query can be linked via parent_query_id to the
        # stale partial one it supersedes — whenever that partial was issued
        self._live: dict[tuple[str, str], list[SubQuery]] = {}
        self._clauses: dict[tuple[str, str], dict[str, set[str]]] = {}

    def reset_utterance(self, utterance_id: str, session_id: str | None = None) -> None:
        """Forget one utterance's state: that of `session_id`, or (no session given) of every session that used
        this utterance id."""
        for d in (self._accumulated, self._previous, self._covered, self._issued, self._live, self._clauses):
            for key in [k for k in d if k[1] == utterance_id and (session_id is None or k[0] == session_id)]:
                d.pop(key, None)

    def reset_session(self, session_id: str) -> None:
        """Forget everything this controller holds for a session (its utterances' text and sub-query state):
        session memory is ephemeral and must not outlive the session."""
        for d in (self._accumulated, self._previous, self._covered, self._issued, self._live, self._clauses):
            for key in [k for k in d if k[0] == session_id]:
                d.pop(key, None)

    async def on_chunk(self, chunk: TranscriptChunk, session: SessionView) -> ControllerDecision:
        uid = chunk.utterance_id
        key = (chunk.session_id, uid)
        previous = self._accumulated.get(key, "")
        text = previous + chunk.text
        self._accumulated[key] = text
        self._previous[key] = previous
        covered = self._covered.setdefault(key, set())

        utterance = text.strip()
        if not utterance and not chunk.is_final:
            return ControllerDecision(Decision.WAIT, TurnKind.NEW_REQUEST, "empty_fragment", 0.0)

        turn_kind = self._classify(utterance, session)

        if turn_kind == TurnKind.CHIT_CHAT:
            return ControllerDecision(Decision.SUPPRESS, TurnKind.CHIT_CHAT, "chit_chat", 1.0)

        if turn_kind == TurnKind.PRESENTATION_ONLY:
            # with no earlier answer there is nothing to reformat: still no corpus search
            return ControllerDecision(Decision.SUPPRESS, TurnKind.PRESENTATION_ONLY,
                                       "presentation_restructure" if session.has_answer()
                                       else "presentation_without_answer", 1.0)

        known_terms = self._corpus_vocab() if self._corpus_vocab is not None else None
        low_df_bigrams = self._corpus_bigrams() if self._corpus_bigrams is not None else None
        confidence = stability.score(utterance, previous, chunk.is_final, known_terms, low_df_bigrams)
        threshold = (self._config.controller.min_stability if chunk.is_final
                      else self._config.controller.provisional_stability)
        if confidence < threshold:
            return ControllerDecision(Decision.WAIT, turn_kind, "intent_unstable", confidence)

        trigger = self._trigger_for(turn_kind, chunk.is_final)
        clause_text: dict[str, str] = {}
        candidates = decompose(utterance, covered, uid, trigger, self._issued.get(key, 0),
                               strip_markers=self._config.controller.strip_discourse_markers,
                               context_carry=self._config.controller.context_carry,
                               clause_sink=clause_text,
                               normalize_numbers=self._config.controller.normalize_spoken_numbers,
                               split_unpunctuated=self._config.controller.split_unpunctuated_questions)
        clauses = self._clauses.setdefault(key, {})
        clauses.update({qid: set(content_tokens(c)) for qid, c in clause_text.items()})

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
        candidates, extra_superseded = self._link_supersession(key, candidates)

        for q in candidates:
            covered |= set(content_tokens(q.intent_label)) | set(content_tokens(q.text))
        self._issued[key] = self._issued.get(key, 0) + len(candidates)

        reason = ("refinement_delta" if turn_kind == TurnKind.REFINEMENT
                   else "multi_intent" if len(candidates) > 1
                   else "intent_stable")
        return ControllerDecision(
            decision=Decision.RETRIEVE, turn_kind=turn_kind, reason=reason,
            stability=confidence, sub_queries=tuple(candidates),
            superseded_query_ids=tuple(extra_superseded),
        )

    async def preview_final(self, utterance_id: str, session_id: str, session: SessionView,
                            pending_text: str = "") -> ControllerDecision:
        """What on_chunk would decide if the utterance ended right now (the
        engine's empty is_final chunk) - or, with `pending_text`, if that text
        (e.g. the ASR's not-yet-committed hypothesis) arrived as one more chunk
        and then the utterance ended. Computed on a snapshot and rolled back:
        the controller's state is exactly as before the call. If exactly
        those words (and no others) arrive, the real passes see the same
        state and session, so they return the same sub-query texts - which
        lets the engine run their retrieval before utterance_end. Returned
        sub_queries = everything either pass would issue."""
        stores = (self._accumulated, self._previous, self._covered, self._issued, self._live, self._clauses)
        missing = object()   # (never deep-copied: a copy would not be `missing`)
        key = (session_id, utterance_id)
        snapshot = [copy.deepcopy(d[key]) if key in d else missing for d in stores]
        try:
            issued: list[SubQuery] = []
            if pending_text:
                first = await self.on_chunk(TranscriptChunk(session_id=session_id, utterance_id=utterance_id,
                                                            timestamp_ms=0, text=pending_text, is_final=False), session)
                if first.decision == Decision.RETRIEVE:
                    issued += first.sub_queries
            final = await self.on_chunk(TranscriptChunk(session_id=session_id, utterance_id=utterance_id,
                                                        timestamp_ms=0, text="", is_final=True), session)
            if not pending_text:
                return final
            if final.decision == Decision.RETRIEVE:
                issued += final.sub_queries
            return ControllerDecision(Decision.RETRIEVE if issued else final.decision, final.turn_kind,
                                      "preview", final.stability, sub_queries=tuple(issued))
        finally:
            for d, v in zip(stores, snapshot):
                if v is missing:
                    d.pop(key, None)
                else:
                    d[key] = v

    def _trigger_for(self, turn_kind: TurnKind, is_final: bool) -> str:
        if turn_kind == TurnKind.REFINEMENT:
            return "refinement"
        return "final" if is_final else "provisional"

    def _classify(self, utterance: str, session: SessionView) -> TurnKind:
        if is_chit_chat(utterance):
            return TurnKind.CHIT_CHAT
        has_prior = session.has_answer()
        if is_presentation_request(utterance):
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

    def _link_supersession(self, key: tuple[str, str], candidates: list[SubQuery],
                            threshold: float = 0.8) -> tuple[list[SubQuery], list[str]]:
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
        cancellation engine.py would otherwise have silently ignored.

        "Supersedes" means the new sub-query's CLAUSE is a grown version of
        the old one's: the old clause's words are (almost) all contained in
        the new clause. Two earlier rules failed: a symmetric overlap score
        (Jaccard >= 0.3) linked different sibling questions that merely
        shared topic words, so the engine cancelled a legitimate question;
        and comparing query texts (after topic words from another clause
        were appended to a partial one) made a stale partial unrecognisable,
        so it was never cancelled. Every still-live sub-query of the
        utterance is a candidate predecessor, not just the latest batch."""
        clauses = self._clauses.get(key, {})
        live = self._live.setdefault(key, [])
        linked = []
        for q in candidates:
            tokens = clauses.get(q.query_id) or set(content_tokens(q.text))
            best_idx, best = None, 0.0
            for i, prior in enumerate(live):
                ptok = clauses.get(prior.query_id) or set(content_tokens(prior.text))
                overlap = len(tokens & ptok) / len(ptok) if ptok else 0.0
                if overlap > best:
                    best, best_idx = overlap, i
            if best_idx is not None and best >= threshold:
                parent = live.pop(best_idx)
                linked.append(SubQuery(query_id=q.query_id, text=q.text, intent_label=q.intent_label,
                                        trigger=q.trigger, utterance_id=q.utterance_id,
                                        parent_query_id=parent.query_id))
            else:
                linked.append(q)
        # A provisional comma-split can fragment one still-growing question
        # into more than one live entry before its "?" arrives (e.g. "Along
        # with diesel engines" / "what engines" from "Along with diesel
        # engines, what engines have overtaken..."). The loop above links
        # each new candidate to at most one prior, so a second, unmatched
        # fragment of the same question would otherwise survive as an
        # orphan and reach synthesis alongside the real answer. Any prior
        # whose own clause is now (almost) fully contained in ANY new
        # candidate is superseded too, even without its own 1:1 link.
        extra_superseded = []
        still_live = []
        for prior in live:
            ptok = clauses.get(prior.query_id) or set(content_tokens(prior.text))
            covered = any((len(ptok & (clauses.get(q.query_id) or set(content_tokens(q.text)))) / len(ptok))
                          >= threshold for q in candidates) if ptok else False
            (extra_superseded if covered else still_live).append(prior)
        live[:] = still_live
        live.extend(linked)
        return linked, [q.query_id for q in extra_superseded]

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
