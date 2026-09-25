"""Event-driven engine: wires Controller -> Retriever -> Synthesizer on one
asyncio event loop (§2 architecture, Task 4 §2.3).

Contract with the outside world (§6.6):
  - input: an asyncio.Queue of envelope dicts {"timestamp_ms", "event_type", "payload"}
  - output: an asyncio.Queue of dicts, each either an "answer_partial" event
    or a "turn_result" record (the guide's §5 record + session/version fields)

The engine never sees ground_truth (the eval loader strips it before the
queue is filled) and never blocks the loop: retrieval runs as background
tasks so speculative retrieval overlaps with incoming speech.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .contracts import (
    ControllerDecision, Decision, RetrievalResult, SubQuery, TranscriptChunk,
    TurnKind, AnswerVersion, Telemetry, NullTelemetry,
)
from .session.store import SessionStore


@dataclass
class _UtteranceState:
    utterance_id: str
    accumulated_text: str = ""
    turn_kind: TurnKind = TurnKind.NEW_REQUEST
    reason: str = ""
    retrieval_required: bool = False
    pending: dict[str, asyncio.Task] = field(default_factory=dict)   # query_id -> task
    query_by_id: dict[str, SubQuery] = field(default_factory=dict)
    retrieval_events: list[dict[str, Any]] = field(default_factory=list)
    completed_results: dict[str, RetrievalResult] = field(default_factory=dict)
    started_ms: int = 0
    suppressed_instruction: str | None = None


class Engine:
    def __init__(self, controller, retriever, synthesizer, llm, config: Config,
                 telemetry: Telemetry | None = None, session_store: SessionStore | None = None):
        self._controller = controller
        self._retriever = retriever
        self._synthesizer = synthesizer
        self._llm = llm
        self._config = config
        self._telemetry = telemetry or NullTelemetry()
        self._sessions = session_store or SessionStore()
        self._utterances: dict[str, _UtteranceState] = {}
        self._request_counter = 0
        self._watchdog_task: asyncio.Task | None = None
        self._closing = asyncio.Event()
        self._anchor_ts_ms = 0
        self._anchor_wall = time.monotonic()

    def _now_ms(self) -> float:
        """Virtual (scenario) clock: anchored to the last envelope's timestamp
        and advanced by real elapsed processing time. --time-scale compresses
        the gaps BETWEEN envelopes (speech), never the engine's own work —
        retrieval and synthesis take the same wall time at any replay speed,
        so scaling them would overstate latency by the replay factor."""
        return self._anchor_ts_ms + (time.monotonic() - self._anchor_wall) * 1000

    def _anchor_clock(self, ts_ms: int) -> None:
        self._anchor_ts_ms = ts_ms
        self._anchor_wall = time.monotonic()

    async def setup(self) -> None:
        """Cold-start work happens here, off the per-turn clock."""
        if hasattr(self._retriever, "setup"):
            await _maybe_await(self._retriever.setup())
        if hasattr(self._synthesizer, "setup"):
            await _maybe_await(self._synthesizer.setup())
        self._watchdog_task = asyncio.create_task(self._loop_lag_watchdog())

    async def teardown(self) -> None:
        self._closing.set()
        if self._watchdog_task is not None:
            await self._watchdog_task

    async def _loop_lag_watchdog(self, tick_ms: float = 20.0) -> None:
        threshold = self._config.engine.loop_lag_threshold_ms / 1000
        tick = tick_ms / 1000
        last = time.monotonic()
        while not self._closing.is_set():
            try:
                await asyncio.wait_for(self._closing.wait(), timeout=tick)
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            drift = (now - last) - tick
            if drift > threshold:
                self._telemetry.emit(
                    "error", component="engine", session_id=None, utterance_id=None,
                    ts_ms=self._now_ms(), where="loop_lag_watchdog", error_type="loop_lag",
                    message=f"tick late by {drift * 1000:.1f}ms",
                )
            last = now

    def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req{self._request_counter}"

    async def run(self, input_queue: "asyncio.Queue[dict]", output_queue: "asyncio.Queue[dict]") -> None:
        await self.setup()
        try:
            while True:
                envelope = await input_queue.get()
                if envelope is None:  # sentinel: stream closed
                    break
                await self._handle_envelope(envelope, output_queue)
        finally:
            await self.teardown()

    async def _handle_envelope(self, envelope: dict, output_queue: "asyncio.Queue[dict]") -> None:
        try:
            event_type = envelope["event_type"]
            payload = envelope.get("payload", {})
            ts_ms = envelope["timestamp_ms"]
        except (KeyError, TypeError):
            self._telemetry.emit("error", component="engine", session_id=None, utterance_id=None,
                                  ts_ms=self._now_ms(), where="handle_envelope",
                                  error_type="malformed_event",
                                  message=f"missing required envelope fields: {envelope!r}")
            return

        self._anchor_clock(ts_ms)

        try:
            if event_type == "session_start":
                await self._on_session_start(payload, ts_ms)
            elif event_type == "transcript_chunk":
                await self._on_transcript_chunk(payload, ts_ms, output_queue)
            elif event_type == "utterance_end":
                await self._on_utterance_end(payload, ts_ms, output_queue)
            elif event_type == "session_end":
                await self._on_session_end(payload, ts_ms)
            else:
                self._telemetry.emit("error", component="engine", session_id=payload.get("session_id"),
                                      utterance_id=payload.get("utterance_id"), ts_ms=self._now_ms(), where="handle_envelope",
                                      error_type="unknown_event_type", message=event_type)
        except Exception as e:  # the engine must never crash on a single bad turn
            self._telemetry.emit("error", component="engine", session_id=payload.get("session_id"),
                                  utterance_id=payload.get("utterance_id"), ts_ms=self._now_ms(), where="handle_envelope",
                                  error_type=type(e).__name__, message=str(e))

    async def _on_session_start(self, payload: dict, ts_ms: int) -> None:
        session_id = payload["session_id"]
        self._sessions.open(session_id)
        self._telemetry.emit("session_started", component="engine", session_id=session_id, ts_ms=ts_ms)

    async def _on_session_end(self, payload: dict, ts_ms: int) -> None:
        session_id = payload.get("session_id")
        self._telemetry.emit("session_ended", component="engine", session_id=session_id, ts_ms=ts_ms)
        if session_id:
            self._sessions.close(session_id)

    async def _on_transcript_chunk(self, payload: dict, ts_ms: int, output_queue) -> None:
        session_id = payload["session_id"] if "session_id" in payload else None
        utterance_id = payload["utterance_id"]
        text = payload.get("text", "")
        session_id = session_id or payload.get("session_id") or self._infer_session(utterance_id)

        self._telemetry.emit("chunk_received", component="engine", session_id=session_id,
                              utterance_id=utterance_id, ts_ms=ts_ms, text_len=len(text), is_final=False)

        state = self._utterances.setdefault(utterance_id, _UtteranceState(utterance_id=utterance_id, started_ms=ts_ms))
        state.accumulated_text += text

        if self._config.engine.mode in ("baseline", "deferred"):
            return  # both wait for utterance_end before doing anything

        chunk = TranscriptChunk(session_id=session_id, utterance_id=utterance_id,
                                 timestamp_ms=ts_ms, text=text, is_final=False)
        await self._dispatch_decision(chunk, session_id, state, ts_ms)

    def _infer_session(self, utterance_id: str) -> str:
        # utterances are per-session; fall back to a stable default if the
        # scenario omits session_id on chunk payloads (session_start carries it).
        for sid in self._sessions._sessions:  # best-effort, ephemeral lookup only
            return sid
        return "default"

    async def _dispatch_decision(self, chunk: TranscriptChunk, session_id: str,
                                  state: _UtteranceState, ts_ms: int) -> None:
        decision_start = time.monotonic()
        try:
            decision: ControllerDecision = await self._controller.on_chunk(chunk, self._sessions.view(session_id))
        except Exception as e:
            self._telemetry.emit("error", component="controller", session_id=session_id,
                                  utterance_id=chunk.utterance_id, ts_ms=ts_ms, where="controller.on_chunk",
                                  error_type=type(e).__name__, message=str(e))
            return
        decision_latency_ms = (time.monotonic() - decision_start) * 1000

        self._telemetry.emit(
            "controller_decision", component="controller", session_id=session_id,
            utterance_id=chunk.utterance_id, ts_ms=ts_ms,
            decision=decision.decision.value, turn_kind=decision.turn_kind.value,
            reason=decision.reason, stability=decision.stability,
            n_sub_queries=len(decision.sub_queries), decision_latency_ms=decision_latency_ms,
        )

        # A trailing WAIT at utterance_end ("no new content") carries no new
        # classification signal and must not clobber a turn_kind an earlier
        # RETRIEVE/SUPPRESS decision already established for this utterance.
        if decision.decision != Decision.WAIT:
            state.turn_kind = decision.turn_kind
            state.reason = decision.reason
        elif not state.reason:
            state.reason = decision.reason

        if decision.decision == Decision.SUPPRESS:
            state.retrieval_required = False
            state.suppressed_instruction = state.accumulated_text
            return

        if decision.decision == Decision.WAIT:
            return

        # RETRIEVE
        state.retrieval_required = True
        self._telemetry.emit(
            "sub_queries_emitted", component="controller", session_id=session_id,
            utterance_id=chunk.utterance_id, ts_ms=ts_ms,
            query_ids=[q.query_id for q in decision.sub_queries],
            queries=[q.text for q in decision.sub_queries],
            trigger=decision.sub_queries[0].trigger if decision.sub_queries else None,
        )

        for extra_id in decision.superseded_query_ids:
            if extra_id in state.pending:
                await self._cancel_task(state, extra_id, session_id, chunk.utterance_id, reason="superseded")
            else:
                state.completed_results.pop(extra_id, None)
                state.query_by_id.pop(extra_id, None)

        for q in decision.sub_queries:
            state.query_by_id[q.query_id] = q
            if q.parent_query_id and q.parent_query_id in state.pending:
                await self._cancel_task(state, q.parent_query_id, session_id, chunk.utterance_id, reason="superseded")
            request_id = self._next_request_id()
            self._telemetry.emit(
                "retrieval_started", component="retriever", session_id=session_id,
                utterance_id=chunk.utterance_id, ts_ms=ts_ms, request_id=request_id,
                query_ids=[q.query_id], mode=self._config.retrieval.mode,
            )
            state.retrieval_events.append({"timestamp_s": ts_ms / 1000, "query": q.text, "trigger": q.trigger})
            task = asyncio.create_task(self._run_retrieval(request_id, [q], session_id, chunk.utterance_id, state))
            state.pending[q.query_id] = task

    async def _run_retrieval(self, request_id: str, queries: list[SubQuery], session_id: str,
                              utterance_id: str, state: _UtteranceState) -> RetrievalResult | None:
        start = time.monotonic()
        try:
            results = await self._retriever.search(queries, k=self._config.retrieval.k)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._telemetry.emit("error", component="retriever", session_id=session_id,
                                  utterance_id=utterance_id, ts_ms=self._now_ms(),
                                  where="retriever.search",
                                  error_type=type(e).__name__, message=str(e))
            return None
        latency_ms = (time.monotonic() - start) * 1000
        for r in results:
            state.completed_results[r.query_id] = r
        prefetch = getattr(self._synthesizer, "prefetch", None)
        if prefetch is not None:
            for r in results:
                q = state.query_by_id.get(r.query_id)
                if q is not None:
                    try:
                        await asyncio.to_thread(prefetch, q, r)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # speculative work must never break the turn
                        self._telemetry.emit("error", component="synthesizer", session_id=session_id,
                                              utterance_id=utterance_id, ts_ms=self._now_ms(),
                                              where="synthesizer.prefetch", error_type=type(e).__name__,
                                              message=str(e))
        self._telemetry.emit(
            "retrieval_completed", component="retriever", session_id=session_id,
            utterance_id=utterance_id, ts_ms=self._now_ms(), request_id=request_id, latency_ms=latency_ms,
            chunk_ids=[e.chunk.chunk_id for r in results for e in r.evidence],
            low_confidence_query_ids=[r.query_id for r in results if r.low_confidence],
        )
        return results[0] if results else None

    async def _cancel_task(self, state: _UtteranceState, query_id: str, session_id: str,
                            utterance_id: str, reason: str) -> None:
        task = state.pending.pop(query_id, None)
        if task is None:
            return
        # "Superseded" means "exclude this evidence from synthesis" — that
        # still applies even if the retrieval already finished by the time
        # the next chunk arrived (the common case for a fast mock/local
        # retriever). Only a still-in-flight task needs an actual cancel();
        # either way the stale result must not reach the synthesizer.
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._telemetry.emit(
            "retrieval_cancelled", component="engine", session_id=session_id,
            utterance_id=utterance_id, ts_ms=self._now_ms(), request_id=None,
            query_ids=[query_id], reason=reason,
        )
        state.completed_results.pop(query_id, None)

    async def _on_utterance_end(self, payload: dict, ts_ms: int, output_queue) -> None:
        utterance_id = payload["utterance_id"]
        session_id = payload.get("session_id") or self._infer_session(utterance_id)
        state = self._utterances.get(utterance_id)
        if state is None:
            state = _UtteranceState(utterance_id=utterance_id, started_ms=ts_ms)
            self._utterances[utterance_id] = state

        self._telemetry.emit("chunk_received", component="engine", session_id=session_id,
                              utterance_id=utterance_id, ts_ms=ts_ms, text_len=0, is_final=True)

        timeout_s = self._config.engine.per_turn_timeout_ms / 1000
        try:
            await asyncio.wait_for(self._finish_turn(state, session_id, ts_ms, output_queue), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._telemetry.emit("error", component="engine", session_id=session_id, utterance_id=utterance_id, ts_ms=ts_ms,
                                  where="finish_turn", error_type="turn_timeout", message="per-turn timeout exceeded")
            await self._emit_degraded_turn_result(state, session_id, ts_ms, output_queue)
        finally:
            self._utterances.pop(utterance_id, None)

    async def _finish_turn(self, state: _UtteranceState, session_id: str, ts_ms: int, output_queue) -> None:
        if self._config.engine.mode == "baseline":
            await self._finish_turn_baseline(state, session_id, ts_ms, output_queue)
            return

        if self._config.engine.mode == "deferred" and state.accumulated_text:
            # Same controller, decomposition, refinement and synthesis as
            # streaming — only the timing differs: the whole utterance is
            # handed over at once, after the user has stopped speaking.
            whole = TranscriptChunk(session_id=session_id, utterance_id=state.utterance_id,
                                     timestamp_ms=ts_ms, text=state.accumulated_text, is_final=False)
            await self._dispatch_decision(whole, session_id, state, ts_ms)
        final_chunk = TranscriptChunk(session_id=session_id, utterance_id=state.utterance_id,
                                       timestamp_ms=ts_ms, text="", is_final=True)
        await self._dispatch_decision(final_chunk, session_id, state, ts_ms)

        if state.pending:
            await asyncio.gather(*state.pending.values(), return_exceptions=True)

        results = [state.completed_results[qid] for qid in state.query_by_id if qid in state.completed_results]
        sub_queries = [state.query_by_id[r.query_id] for r in results]

        try:
            if state.turn_kind == TurnKind.PRESENTATION_ONLY:
                answer = await self._synthesizer.restructure(session_id, state.accumulated_text)
            elif state.turn_kind == TurnKind.CHIT_CHAT:
                answer = None
            elif state.turn_kind == TurnKind.REFINEMENT and self._sessions.view(session_id).has_answer():
                answer = await self._synthesizer.refine(session_id, state.accumulated_text, sub_queries, results)
            else:
                answer = await self._synthesizer.synthesize(session_id, state.accumulated_text, sub_queries, results)
        except Exception as e:
            self._telemetry.emit("error", component="synthesizer", session_id=session_id,
                                  utterance_id=state.utterance_id, ts_ms=ts_ms, where="synthesizer", error_type=type(e).__name__,
                                  message=str(e))
            await self._emit_degraded_turn_result(state, session_id, ts_ms, output_queue)
            return

        if answer is not None:
            self._sessions.record_answer(session_id, answer)
            self._emit_answer_telemetry(answer, session_id, state.utterance_id, results)

        await self._emit_turn_result(state, session_id, ts_ms, answer, output_queue)

    async def _finish_turn_baseline(self, state: _UtteranceState, session_id: str, ts_ms: int, output_queue) -> None:
        text = state.accumulated_text.strip()
        sub_query = SubQuery(query_id=f"{state.utterance_id}.baseline", text=text,
                              intent_label=text[:40], trigger="final", utterance_id=state.utterance_id)
        request_id = self._next_request_id()
        self._telemetry.emit("retrieval_started", component="retriever", session_id=session_id,
                              utterance_id=state.utterance_id, ts_ms=ts_ms, request_id=request_id,
                              query_ids=[sub_query.query_id], mode=self._config.retrieval.mode)
        try:
            results = await self._retriever.search([sub_query], k=self._config.retrieval.k)
        except Exception as e:
            self._telemetry.emit("error", component="retriever", session_id=session_id,
                                  utterance_id=state.utterance_id, ts_ms=ts_ms, where="retriever.search",
                                  error_type=type(e).__name__, message=str(e))
            await self._emit_degraded_turn_result(state, session_id, ts_ms, output_queue)
            return
        self._telemetry.emit("retrieval_completed", component="retriever", session_id=session_id,
                              utterance_id=state.utterance_id, ts_ms=self._now_ms(), request_id=request_id,
                              latency_ms=results[0].latency_ms if results else 0.0,
                              chunk_ids=[e.chunk.chunk_id for r in results for e in r.evidence],
                              low_confidence_query_ids=[r.query_id for r in results if r.low_confidence])
        state.retrieval_events.append({"timestamp_s": ts_ms / 1000, "query": text, "trigger": "final"})
        try:
            answer = await self._synthesizer.synthesize(session_id, text, [sub_query], results)
        except Exception as e:
            self._telemetry.emit("error", component="synthesizer", session_id=session_id,
                                  utterance_id=state.utterance_id, ts_ms=ts_ms, where="synthesizer",
                                  error_type=type(e).__name__, message=str(e))
            await self._emit_degraded_turn_result(state, session_id, ts_ms, output_queue)
            return
        self._sessions.record_answer(session_id, answer)
        self._emit_answer_telemetry(answer, session_id, state.utterance_id, results)
        await self._emit_turn_result(state, session_id, ts_ms, answer, output_queue)

    def _emit_answer_telemetry(self, answer: AnswerVersion, session_id: str, utterance_id: str,
                                results: list[RetrievalResult]) -> None:
        now = self._now_ms()
        self._telemetry.emit(
            "answer_version_created", component="synthesizer", session_id=session_id, utterance_id=utterance_id,
            ts_ms=now, version=answer.version, parent_version=answer.parent_version,
            change_kind=answer.change_kind, citations=answer.citations, n_claims=len(answer.claims),
            retriever_calls_for_version=len(results),
        )
        n_supported = sum(1 for c in answer.claims if c.supported)
        fabricated = [c for cl in answer.claims for c in cl.citations
                       if self._retriever.get_chunk_by_citation(c) is None] \
            if hasattr(self._retriever, "get_chunk_by_citation") else []
        self._telemetry.emit(
            "grounding_checked", component="synthesizer", session_id=session_id, utterance_id=utterance_id,
            ts_ms=now, version=answer.version, n_claims=len(answer.claims), n_supported=n_supported,
            fabricated_citations=fabricated,
        )
        if answer.uncertainty:
            self._telemetry.emit(
                "uncertainty_flagged", component="synthesizer", session_id=session_id, utterance_id=utterance_id,
                ts_ms=now, version=answer.version, unsupported_intents=answer.uncertainty,
            )

    async def _emit_turn_result(self, state: _UtteranceState, session_id: str, ts_ms: int,
                                 answer: AnswerVersion | None, output_queue) -> None:
        record = {
            "kind": "turn_result",
            "session_id": session_id,
            "utterance_id": state.utterance_id,
            "retrieval_events": state.retrieval_events,
            "sub_queries": [q.text for q in state.query_by_id.values()],
            "answer": answer.text if answer else "",
            "citations": answer.citations if answer else [],
            "uncertainty": answer.uncertainty if answer else None,
            "answer_version": answer.version if answer else None,
            "parent_version": answer.parent_version if answer else None,
            "retrieval_required": state.retrieval_required,
            "reason": state.reason,
        }
        # ts_ms here is the virtual clock AFTER retrieval+synthesis, not the
        # envelope's raw timestamp — otherwise the trace would show zero
        # end-to-end latency regardless of how long the turn actually took.
        self._telemetry.emit("output_emitted", component="engine", session_id=session_id,
                              utterance_id=state.utterance_id, ts_ms=self._now_ms(), kind="turn_result",
                              version=answer.version if answer else None)
        await output_queue.put(record)

    async def _emit_degraded_turn_result(self, state: _UtteranceState, session_id: str, ts_ms: int, output_queue) -> None:
        record = {
            "kind": "turn_result",
            "session_id": session_id,
            "utterance_id": state.utterance_id,
            "retrieval_events": state.retrieval_events,
            "sub_queries": [q.text for q in state.query_by_id.values()],
            "answer": "I'm not able to complete this right now.",
            "citations": [],
            "uncertainty": "internal error or timeout prevented a grounded answer",
            "answer_version": None,
            "parent_version": None,
            "retrieval_required": state.retrieval_required,
            "reason": "degraded",
        }
        self._telemetry.emit("output_emitted", component="engine", session_id=session_id,
                              utterance_id=state.utterance_id, ts_ms=self._now_ms(), kind="turn_result", version=None)
        await output_queue.put(record)


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value
