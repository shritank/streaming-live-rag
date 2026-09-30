"""A deterministic, offline stand-in for an LLM, for integration tests only.

C's FakeLLM returns "{}", which the generative synthesizer correctly treats as
"no claims", so with it every turn would become a clarification and the
refine / restructure paths would never run end to end. This double reads the
SOURCES block out of the prompt and answers in the JSON shapes the generative
synthesizer parses, citing only ids the prompt actually contains.

It lives under tests/ on purpose. It says nothing about answer *quality*: it
just copies a source sentence, so scoring it against gold answers would repeat
the self-grading mistake the audit found in the gate suite.
"""
from __future__ import annotations

import json
import re
from typing import Any

from streaming_rag.contracts import LLMResponse

_SOURCE_RE = re.compile(r"\[(?P<citation>[^\]]+)\] \(chunk [^)]+\)\n<<<\n(?P<text>.*?)\n>>>", re.DOTALL)
_SUBQ_RE = re.compile(r"^- (?P<query_id>\S+): (?P<text>.+)$", re.MULTILINE)


def _overlap(a: str, b: str) -> float:
    left = set(re.findall(r"[a-z]{3,}", a.lower()))
    right = set(re.findall(r"[a-z]{3,}", b.lower()))
    return len(left & right) / len(left) if left else 0.0


class OfflineClaimsLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def purposes(self) -> list[str]:
        return [c["purpose"] for c in self.calls]

    async def complete(self, *, purpose: str, system: str, prompt: str,
                       json_mode: bool = False, max_tokens: int = 512) -> LLMResponse:
        self.calls.append({"purpose": purpose, "prompt": prompt})
        text = json.dumps(self._answer(purpose, prompt), ensure_ascii=False)
        return LLMResponse(text=text, model="offline-claims", tokens_in=len(prompt) // 4,
                           tokens_out=len(text) // 4, latency_ms=0.0)

    def _answer(self, purpose: str, prompt: str) -> dict[str, Any]:
        if purpose in ("synthesis", "refinement_claims"):
            return {"claims": self._claims(prompt)}
        if purpose == "delta_decisions":
            rows = re.findall(r"^(\d+)\. ", prompt, re.MULTILINE)
            return {"decisions": [{"index": int(i), "action": "keep"} for i in rows]}
        if purpose == "restructure":
            match = re.search(r"CURRENT ANSWER:\n(.*?)\n\nITS CLAIMS", prompt, re.DOTALL)
            return {"text": match.group(1).strip() if match else ""}
        if purpose == "clarification":
            return {"question": "Could you give me more detail to search on?"}
        return {}

    def _claims(self, prompt: str) -> list[dict[str, Any]]:
        sources = [(m["citation"], m["text"].strip()) for m in _SOURCE_RE.finditer(prompt)]
        queries = [(m["query_id"], m["text"]) for m in _SUBQ_RE.finditer(prompt)]
        claims, used = [], set()
        for query_id, query_text in queries or [(None, "")]:
            if not sources:
                break
            citation, text = max(sources, key=lambda s: _overlap(query_text, s[1]))
            if citation in used:
                continue
            used.add(citation)
            claims.append({"text": text.split(". ")[0].rstrip("."), "citations": [citation],
                           "sub_query_id": query_id})
        return claims
