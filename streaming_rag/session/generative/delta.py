"""The answer delta engine — refine, don't restart.

When a late constraint arrives we already hold the previous answer as claims
with their sources. Rather than re-asking the whole question, each existing
claim is judged once: keep, amend or retract. New claims are generated only for
what the constraint introduces.

If the model call fails or returns nonsense, a deterministic heuristic takes
over (retract claims whose numbers now conflict, keep the rest). G5 is a
continuity gate, so degrading to a weaker-but-correct update beats failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from ...contracts import AnswerVersion, Claim, EvidenceChunk, LLMClient, SubQuery
from . import prompts
from .grounding import _content_words, _first_json_blob, _numbers

Action = Literal["keep", "amend", "retract"]


@dataclass
class ClaimDecision:
    index: int
    action: Action
    text: str                  # unchanged for keep, rewritten for amend
    citations: list[str]
    reason: str = ""


@dataclass
class DeltaPlan:
    decisions: list[ClaimDecision]
    used_fallback: bool = False

    def counts(self) -> dict[str, int]:
        tally = {"keep": 0, "amend": 0, "retract": 0}
        for decision in self.decisions:
            tally[decision.action] += 1
        return tally

    def surviving_claims(self) -> list[Claim]:
        return [Claim(text=d.text, citations=list(d.citations))
                for d in self.decisions if d.action in ("keep", "amend")]


class DeltaEngine:
    """Decides what happens to each existing claim under a new constraint."""

    def __init__(self, llm: LLMClient, config: dict[str, Any]) -> None:
        self.llm = llm
        self.cfg = config

    async def plan(self, previous: AnswerVersion, utterance: str,
                   delta_queries: list[SubQuery],
                   evidence: list[EvidenceChunk], max_chars: int) -> DeltaPlan:
        if not previous.claims:
            return DeltaPlan(decisions=[])

        prompt = prompts.build_delta_prompt(previous, utterance, delta_queries,
                                            evidence, max_chars)
        try:
            response = await self.llm.complete(
                purpose="delta_decisions", system=prompts.DELTA_SYSTEM,
                prompt=prompt, json_mode=True, max_tokens=800,
            )
            decisions = self._parse(response.text, previous)
            if decisions:
                return DeltaPlan(decisions=decisions)
        except Exception:  # noqa: BLE001 - any model failure falls through
            pass
        return DeltaPlan(decisions=self._fallback(previous, utterance, evidence),
                         used_fallback=True)

    # ----- parsing -----
    def _parse(self, raw: str, previous: AnswerVersion) -> list[ClaimDecision]:
        blob = _first_json_blob(raw)
        if not blob:
            return []
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            return []
        rows = payload.get("decisions") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []

        seen: dict[int, ClaimDecision] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                index = int(row.get("index", -1))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(previous.claims) or index in seen:
                continue
            action = str(row.get("action", "keep")).lower()
            if action not in ("keep", "amend", "retract"):
                action = "keep"
            original = previous.claims[index]
            text = str(row.get("text") or original.text).strip() or original.text
            citations = row.get("citations")
            if not isinstance(citations, list) or not citations:
                citations = list(original.citations)
            seen[index] = ClaimDecision(
                index=index, action=action,  # type: ignore[arg-type]
                text=original.text if action == "keep" else text,
                citations=[str(c) for c in citations],
                reason=str(row.get("reason", ""))[:200],
            )

        # Claims the model forgot about are kept untouched: silence must not
        # delete a grounded claim.
        for index, claim in enumerate(previous.claims):
            seen.setdefault(index, ClaimDecision(
                index=index, action="keep", text=claim.text,
                citations=list(claim.citations), reason="not_mentioned_by_model",
            ))
        return [seen[i] for i in sorted(seen)]

    # ----- deterministic fallback -----
    def _fallback(self, previous: AnswerVersion, utterance: str,
                  evidence: list[EvidenceChunk]) -> list[ClaimDecision]:
        """Retract claims the new constraint contradicts; keep everything else.

        "Topical" is judged against the new utterance *and* the evidence the
        delta retrieved, because a constraint like "make it 60 people" shares no
        vocabulary with the claim it invalidates ("Hall Aurora seats 30
        attendees") while the delta evidence does.
        """
        new_numbers = _numbers(utterance)
        constraint_words = _content_words(utterance)
        for ev in evidence:
            constraint_words |= _content_words(ev.chunk.text)
        decisions: list[ClaimDecision] = []
        for index, claim in enumerate(previous.claims):
            claim_numbers = _numbers(claim.text)
            topical = bool(constraint_words & _content_words(claim.text))
            conflicts = bool(new_numbers and claim_numbers and topical
                             and not (new_numbers & claim_numbers))
            decisions.append(ClaimDecision(
                index=index,
                action="retract" if conflicts else "keep",
                text=claim.text,
                citations=list(claim.citations),
                reason="heuristic_number_conflict" if conflicts else "heuristic_keep",
            ))
        return decisions
