from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol

# ---------- controller ----------
class Decision(str, Enum):
    WAIT = "wait"            # intent not yet stable; do nothing
    RETRIEVE = "retrieve"    # run retrieval for decision.sub_queries
    SUPPRESS = "suppress"    # retrieval_required = false (formatting/chit-chat/answerable from session)

class TurnKind(str, Enum):
    NEW_REQUEST = "new_request"
    REFINEMENT = "refinement"                # late constraint on the current topic
    PRESENTATION_ONLY = "presentation_only"  # reformat / shorten / translate the previous answer
    CHIT_CHAT = "chit_chat"                  # greetings, thanks, no information need

Trigger = Literal["provisional", "multi_intent", "refinement", "final"]

@dataclass(frozen=True)
class TranscriptChunk:
    session_id: str
    utterance_id: str
    timestamp_ms: int
    text: str        # only the NEW fragment
    is_final: bool   # True on the utterance_end event (text may be "")

@dataclass(frozen=True)
class SubQuery:
    query_id: str                  # unique within the session, e.g. "u1.q2"
    text: str                      # search-ready query string
    intent_label: str              # short label, e.g. "cancellation policy"
    trigger: Trigger
    utterance_id: str
    parent_query_id: str | None = None   # set when this is a refinement delta of an earlier query

@dataclass(frozen=True)
class ControllerDecision:
    decision: Decision
    turn_kind: TurnKind
    reason: str                    # snake_case, e.g. "intent_unstable", "presentation_restructure"
    stability: float               # 0..1 confidence that the intent is settled
    sub_queries: tuple[SubQuery, ...] = ()   # non-empty iff decision == RETRIEVE

# ---------- retrieval ----------
@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # "<doc_id> §<section> #<n>", e.g. "Doc_12 §2 #3"
    doc_id: str          # "Doc_12"
    section: str         # "2"  (or "2.1")
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:          # what appears in answers: "Doc_12 §2"
        return f"{self.doc_id} §{self.section}"

@dataclass(frozen=True)
class EvidenceChunk:
    chunk: Chunk
    score: float                       # fused score, higher is better
    query_ids: tuple[str, ...]         # sub-queries that retrieved this chunk
    dense_rank: int | None = None
    sparse_rank: int | None = None

@dataclass(frozen=True)
class RetrievalResult:
    query_id: str
    evidence: tuple[EvidenceChunk, ...]   # ranked, deduplicated
    latency_ms: float
    low_confidence: bool                  # best score under the calibrated threshold → likely not in corpus

class Retriever(Protocol):
    async def search(self, queries: list[SubQuery], k: int = 5) -> list[RetrievalResult]: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

# ---------- session & synthesis ----------
@dataclass
class Claim:
    text: str
    citations: list[str]              # ["Doc_12 §2"]
    supported: bool | None = None     # set by the grounding verifier

@dataclass
class AnswerVersion:
    version: int                                   # 1, 2, 3 …
    parent_version: int | None
    change_kind: Literal["initial", "refinement", "restructure", "clarification"]
    text: str
    claims: list[Claim]
    citations: list[str]                           # de-duplicated union of claim citations
    evidence_ids: list[str]                        # chunk_ids actually used
    sub_queries: list[SubQuery]
    uncertainty: str | None
    created_ms: int

class SessionView(Protocol):                       # read-only view given to the controller
    session_id: str
    def has_answer(self) -> bool: ...
    def latest_answer(self) -> AnswerVersion | None: ...
    def topic_summary(self) -> str: ...            # short text of the current topic ("" if none)
    def prior_sub_queries(self) -> list[SubQuery]: ...

class Synthesizer(Protocol):
    async def synthesize(self, session_id: str, utterance: str,
                         sub_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion: ...
    async def refine(self, session_id: str, utterance: str,
                     delta_queries: list[SubQuery], results: list[RetrievalResult]) -> AnswerVersion: ...
    async def restructure(self, session_id: str, instruction: str) -> AnswerVersion: ...

# ---------- cross-cutting ----------
@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float

class LLMClient(Protocol):
    async def complete(self, *, purpose: str, system: str, prompt: str,
                       json_mode: bool = False, max_tokens: int = 512) -> LLMResponse: ...

class Telemetry(Protocol):
    def emit(self, event: str, **fields: Any) -> None: ...

class NullTelemetry:
    def emit(self, event: str, **fields: Any) -> None:
        pass
