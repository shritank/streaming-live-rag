"""Fits build A's LLM synthesizer into build C's engine.

Three mismatches are bridged here, and each one is a way the merge could fail
silently rather than loudly:

1. **Two session stores.** A's synthesizer keeps its own store; C's engine
   keeps another and wipes it at session end. Holding state in both would let
   A's copy outlive the session (a session-bound-state violation). Instead this
   adapter rebuilds A's working store from C's on every call, so C's store is
   the only record and `close()` still forgets everything.
2. **Duplicate telemetry.** C's engine emits `answer_version_created` and
   `grounding_checked` itself, with the utterance id and virtual clock
   attached. A's synthesizer emits the same events without them. Forwarding
   both would put two copies of each event in the trace, so only A's `error`
   events are passed through.
3. **Constructor shape.** C builds synthesizers as
   `(retriever, session_store, config, llm, telemetry)`.
"""

from __future__ import annotations

from typing import Any

from ...contracts import AnswerVersion, LLMClient, NullTelemetry, RetrievalResult, SubQuery, Telemetry
from ..store import SessionStore as EngineSessionStore
from .grounding import GroundingVerifier
from .settings import load_config
from .store import SessionStore as WorkingStore
from .synthesizer import GroundedSynthesizer

# Events C's engine already emits for every turn, attributed to the turn.
_ENGINE_OWNED_EVENTS = frozenset({"answer_version_created", "grounding_checked", "uncertainty_flagged"})


class _ForwardErrorsOnly:
    """Passes the synthesizer's errors through; drops events the engine owns."""

    def __init__(self, sink: Telemetry) -> None:
        self._sink = sink

    def emit(self, event: str, **fields: Any) -> None:
        if event in _ENGINE_OWNED_EVENTS:
            return
        fields.setdefault("component", "synthesizer")
        self._sink.emit(event, **fields)


def settings_from_config(config: Any) -> dict[str, Any]:
    """Map C's Config onto the generative package's settings."""
    synthesis = getattr(config, "synthesis", None)
    overrides: dict[str, Any] = {}
    checker = getattr(synthesis, "grounding_checker", None)
    if checker:
        overrides["grounding"] = {"checker": checker}
    if synthesis is not None and not getattr(synthesis, "grounding_check", True):
        # Verification still strips invented ids; it simply stops dropping
        # claims on weak support (used for ablation runs).
        overrides.setdefault("grounding", {})["drop_unsupported"] = False
    return load_config(overrides)


class GenerativeSynthesizer:
    """contracts.Synthesizer backed by an LLM, built like C's synthesizers."""

    def __init__(self, retriever: Any, session_store: EngineSessionStore, config: Any,
                 llm: LLMClient | None = None, telemetry: Telemetry | None = None) -> None:
        if llm is None:
            raise ValueError("synthesis.mode=generative needs an LLM client")
        self._sessions = session_store
        self._llm = llm
        self._settings = settings_from_config(config)
        self._telemetry = _ForwardErrorsOnly(telemetry or NullTelemetry())
        self._verifier = GroundingVerifier(config=self._settings, telemetry=self._telemetry, llm=llm)

    # ---------- Synthesizer protocol ----------
    async def synthesize(self, session_id: str, utterance: str,
                         sub_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        synth = self._working_synthesizer(session_id)
        answer = await synth.synthesize(session_id, utterance, sub_queries, results)
        self._keep_evidence(session_id, results)
        return answer

    async def refine(self, session_id: str, utterance: str,
                     delta_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion:
        synth = self._working_synthesizer(session_id)
        answer = await synth.refine(session_id, utterance, delta_queries, results)
        self._keep_evidence(session_id, results)
        return answer

    async def restructure(self, session_id: str, instruction: str) -> AnswerVersion:
        # Unlike C's extractive synthesizer, this never raises when there is
        # nothing to restructure: it returns a clarification version instead.
        return await self._working_synthesizer(session_id).restructure(session_id, instruction)

    # ---------- internals ----------
    def _working_synthesizer(self, session_id: str) -> GroundedSynthesizer:
        """A fresh A-side store, replayed from the engine's store."""
        answers, sub_queries, evidence = self._sessions.history(session_id)
        working = WorkingStore(config=self._settings)
        working.open(session_id)
        working.add_evidence(session_id, evidence)
        working.add_sub_queries(session_id, sub_queries)
        for answer in answers:
            working.add_version(session_id, answer)
        return GroundedSynthesizer(store=working, llm=self._llm, verifier=self._verifier,
                                   telemetry=self._telemetry, config=self._settings)

    def _keep_evidence(self, session_id: str, results: list[RetrievalResult]) -> None:
        """Record this turn's evidence in the engine's store.

        A refinement re-verifies the claims it keeps against earlier evidence,
        so that evidence has to survive the turn. It lives in the engine's
        store, which the engine wipes at session end.
        """
        if results:
            self._sessions.add_evidence(session_id, results)
