"""Ephemeral, session-bound state.

Hard rules this file exists to enforce (project context §4):

* memory is in-process and scoped to one conversation;
* `close()` wipes everything, so nothing survives a session;
* answer versions are append-only — a refinement never mutates its parent.

Nothing here writes to disk.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ...contracts import AnswerVersion, EvidenceChunk, RetrievalResult, SubQuery
from .settings import get, load_config


class UnknownSessionError(KeyError):
    """Raised when a session id was never opened, or was already closed."""


@dataclass
class SessionState:
    """Everything the system remembers about one live conversation."""

    session_id: str
    created_ms: int
    turns: list[str] = field(default_factory=list)
    versions: list[AnswerVersion] = field(default_factory=list)
    # chunk_id -> EvidenceChunk; the cache that makes refinement cheap, because
    # a late constraint only has to retrieve the delta.
    evidence: dict[str, EvidenceChunk] = field(default_factory=dict)
    sub_queries: list[SubQuery] = field(default_factory=list)
    low_confidence_query_ids: set[str] = field(default_factory=set)

    def citations(self) -> set[str]:
        return {ev.chunk.citation for ev in self.evidence.values()}


class _SessionViewImpl:
    """Read-only window onto a session (contracts.SessionView).

    Handed to the controller (Task 2) and to the synthesizer so neither can
    mutate state by accident.
    """

    def __init__(self, state: SessionState) -> None:
        self._state = state
        self.session_id = state.session_id

    def has_answer(self) -> bool:
        return bool(self._state.versions)

    def latest_answer(self) -> AnswerVersion | None:
        return self._state.versions[-1] if self._state.versions else None

    def topic_summary(self) -> str:
        """A cheap, LLM-free description of what this session is about."""
        if not self._state.turns:
            return ""
        labels = [sq.intent_label for sq in self._state.sub_queries if sq.intent_label]
        first_turn = self._state.turns[0].strip()
        if len(first_turn) > 160:
            first_turn = first_turn[:157].rstrip() + "..."
        if labels:
            # dict.fromkeys keeps first-seen order while removing duplicates
            return f"{first_turn} [{', '.join(dict.fromkeys(labels))}]"
        return first_turn

    def prior_sub_queries(self) -> list[SubQuery]:
        return list(self._state.sub_queries)


class SessionStore:
    """In-memory store of live sessions (project context §6.3)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.cfg = config or load_config()
        self._sessions: dict[str, SessionState] = {}

    # ----- lifecycle -----
    def open(self, session_id: str) -> None:
        if session_id in self._sessions:
            return
        self._sessions[session_id] = SessionState(
            session_id=session_id, created_ms=int(time.time() * 1000)
        )

    def close(self, session_id: str) -> None:
        """Forget the session completely. Safe to call twice."""
        state = self._sessions.pop(session_id, None)
        if state is None:
            return
        # Drop references eagerly so large evidence caches are collectable even
        # if something else still holds the SessionState object.
        state.turns.clear()
        state.versions.clear()
        state.evidence.clear()
        state.sub_queries.clear()
        state.low_confidence_query_ids.clear()

    def is_open(self, session_id: str) -> bool:
        return session_id in self._sessions

    def session_ids(self) -> list[str]:
        return list(self._sessions)

    # ----- reads -----
    def view(self, session_id: str) -> _SessionViewImpl:
        return _SessionViewImpl(self._state(session_id))

    def state(self, session_id: str) -> SessionState:
        """Internal accessor used by the synthesizer; not part of SessionView."""
        return self._state(session_id)

    def evidence_for(self, session_id: str,
                     chunk_ids: Iterable[str] | None = None) -> dict[str, EvidenceChunk]:
        """Cached evidence, optionally narrowed to specific chunk ids."""
        cache = self._state(session_id).evidence
        if chunk_ids is None:
            return dict(cache)
        return {cid: cache[cid] for cid in chunk_ids if cid in cache}

    def evidence_by_citation(self, session_id: str) -> dict[str, list[EvidenceChunk]]:
        """Citation label -> every cached chunk carrying it."""
        grouped: dict[str, list[EvidenceChunk]] = {}
        for ev in self._state(session_id).evidence.values():
            grouped.setdefault(ev.chunk.citation, []).append(ev)
        return grouped

    def latest_version(self, session_id: str) -> AnswerVersion | None:
        versions = self._state(session_id).versions
        return versions[-1] if versions else None

    # ----- writes -----
    def add_turn(self, session_id: str, text: str) -> None:
        state = self._state(session_id)
        state.turns.append(text)
        self._trim(state.turns, int(get(self.cfg, "session.max_turns", 50)))

    def add_evidence(self, session_id: str, results: list[RetrievalResult]) -> None:
        """Merge retrieval results into the session cache.

        When the same chunk arrives again the higher-scoring copy wins and the
        query_ids are merged, so the cache records every sub-query that found it.
        """
        state = self._state(session_id)
        for result in results:
            if result.low_confidence:
                state.low_confidence_query_ids.add(result.query_id)
            else:
                state.low_confidence_query_ids.discard(result.query_id)
            for ev in result.evidence:
                existing = state.evidence.get(ev.chunk.chunk_id)
                if existing is None:
                    state.evidence[ev.chunk.chunk_id] = ev
                    continue
                merged_ids = tuple(dict.fromkeys(existing.query_ids + ev.query_ids))
                winner = ev if ev.score > existing.score else existing
                state.evidence[ev.chunk.chunk_id] = EvidenceChunk(
                    chunk=winner.chunk, score=winner.score, query_ids=merged_ids,
                    dense_rank=winner.dense_rank, sparse_rank=winner.sparse_rank,
                )

    def add_sub_queries(self, session_id: str, sub_queries: Iterable[SubQuery]) -> None:
        state = self._state(session_id)
        known = {sq.query_id for sq in state.sub_queries}
        for sq in sub_queries:
            if sq.query_id not in known:
                state.sub_queries.append(sq)
                known.add(sq.query_id)

    def add_version(self, session_id: str, version: AnswerVersion) -> None:
        """Append a version. Versions are immutable once stored."""
        state = self._state(session_id)
        if state.versions and version.version != state.versions[-1].version + 1:
            raise ValueError(
                f"version {version.version} does not follow "
                f"{state.versions[-1].version} in session {session_id}"
            )
        state.versions.append(version)
        self._trim(state.versions, int(get(self.cfg, "session.max_versions", 50)))

    def next_version_number(self, session_id: str) -> int:
        latest = self.latest_version(session_id)
        return 1 if latest is None else latest.version + 1

    # ----- internals -----
    def _state(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise UnknownSessionError(
                f"session {session_id!r} is not open (never opened, or closed)"
            ) from exc

    @staticmethod
    def _trim(items: list[Any], limit: int) -> None:
        while len(items) > limit:
            items.pop(0)
