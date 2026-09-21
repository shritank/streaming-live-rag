"""Grounding verification (G4).

Two distinct failures are checked separately, because they have different
consequences:

  fabricated citation - the cited `Doc_ID §Section` does not exist in the
                        corpus at all (the guide's `[Doc_999]` pitfall).
                        This must be zero; such claims are dropped outright.
  unsupported claim   - the citation resolves, but the chunk does not
                        actually contain the asserted content. The claim is
                        marked unsupported and surfaced as uncertainty rather
                        than presented as fact.

Support is measured as the share of the claim's content tokens that appear
in the cited chunk. Extractive synthesis makes this near-perfect by
construction; the check exists to catch an LLM-phrased answer drifting off
its evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts import Claim
from ..retrieval.text import content_tokens

SUPPORT_THRESHOLD = 0.6


@dataclass
class GroundingReport:
    claims: list[Claim]
    fabricated_citations: list[str]
    n_supported: int

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def support_rate(self) -> float:
        return self.n_supported / self.n_claims if self.claims else 1.0


def claim_support(claim_text: str, chunk_text: str) -> float:
    tokens = content_tokens(claim_text)
    if not tokens:
        return 0.0
    chunk_tokens = set(content_tokens(chunk_text))
    return sum(1 for t in tokens if t in chunk_tokens) / len(tokens)


def verify(claims: list[Claim], resolve_citation) -> GroundingReport:
    """`resolve_citation(citation) -> Chunk | None` proves the citation exists.
    Returns a report whose claims have `supported` populated and whose
    fabricated citations have been stripped from the claims."""
    fabricated: list[str] = []
    kept: list[Claim] = []
    n_supported = 0

    for claim in claims:
        real_citations = []
        best_support = 0.0
        for citation in claim.citations:
            chunk = resolve_citation(citation)
            if chunk is None:
                fabricated.append(citation)
                continue
            real_citations.append(citation)
            best_support = max(best_support, claim_support(claim.text, chunk.text))

        if not real_citations:
            continue  # every citation was fabricated: the claim cannot stand

        supported = best_support >= SUPPORT_THRESHOLD
        if supported:
            n_supported += 1
        kept.append(Claim(text=claim.text, citations=real_citations, supported=supported))

    return GroundingReport(claims=kept, fabricated_citations=fabricated, n_supported=n_supported)
