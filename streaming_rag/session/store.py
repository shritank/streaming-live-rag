"""Ephemeral, session-bound state (§6.3, §4 hard rule "session-bound state").

No cross-session profiling: state lives only in this process's memory for
the lifetime of the session and is wiped on close(). Nothing here is ever
persisted to disk; only telemetry (elsewhere) writes to disk, and telemetry
carries no cross-session identifiers beyond the session_id itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import AnswerVersion, RetrievalResult, SubQuery


@dataclass
class _SessionState:
    session_id: str
    answers: list[AnswerVersion] = field(default_factory=list)
    sub_queries: list[SubQuery] = field(default_factory=list)
    evidence: list[RetrievalResult] = field(default_factory=list)


class InMemorySessionView:
    """Read-only view handed to the controller. Wraps a live _SessionState
    reference so it always reflects the current state without copying."""

    def __init__(self, state: _SessionState):
        self._state = state
        self.session_id = state.session_id

    def has_answer(self) -> bool:
        return len(self._state.answers) > 0

    def latest_answer(self) -> AnswerVersion | None:
        return self._state.answers[-1] if self._state.answers else None

    def topic_summary(self) -> str:
        latest = self.latest_answer()
        if latest is None:
            return ""
        labels = [q.intent_label for q in latest.sub_queries]
        return "; ".join(labels[:3])

    def prior_sub_queries(self) -> list[SubQuery]:
        return list(self._state.sub_queries)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, _SessionState] = {}

    def open(self, session_id: str) -> None:
        self._sessions[session_id] = _SessionState(session_id=session_id)

    def view(self, session_id: str) -> InMemorySessionView:
        if session_id not in self._sessions:
            self.open(session_id)
        return InMemorySessionView(self._sessions[session_id])

    def add_evidence(self, session_id: str, results: list[RetrievalResult]) -> None:
        state = self._sessions.setdefault(session_id, _SessionState(session_id=session_id))
        state.evidence.extend(results)

    def record_answer(self, session_id: str, answer: AnswerVersion) -> None:
        state = self._sessions.setdefault(session_id, _SessionState(session_id=session_id))
        state.answers.append(answer)
        for q in answer.sub_queries:
            if q not in state.sub_queries:
                state.sub_queries.append(q)

    def close(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def history(self, session_id: str) -> tuple[list[AnswerVersion], list[SubQuery], list[RetrievalResult]]:
        """Copies of a session's answers, sub-queries and evidence, oldest first.

        Read-only: callers get new lists, so nothing they do can mutate the
        session. Used by the generative synthesizer, which rebuilds its working
        state from here on every turn so this store stays the single source of
        truth and close() still forgets everything.
        """
        state = self._sessions.get(session_id)
        if state is None:
            return [], [], []
        return list(state.answers), list(state.sub_queries), list(state.evidence)
