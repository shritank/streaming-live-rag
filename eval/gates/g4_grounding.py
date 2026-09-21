"""G4 — factual grounding: >= 85% citation support, zero fabricated document
IDs, aggregated from every `grounding_checked` telemetry event in the trace.
No ground_truth needed: grounding is checked against the corpus itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class G4Result:
    n_claims: int
    n_supported: int
    fabricated: list[str]

    @property
    def support_rate(self) -> float:
        return self.n_supported / self.n_claims if self.n_claims else 1.0

    @property
    def passed(self) -> bool:
        return self.support_rate >= 0.85 and len(self.fabricated) == 0


def score_g4(trace_events: list[dict], ground_truth: dict | None = None) -> G4Result:
    n_claims = n_supported = 0
    fabricated: list[str] = []
    for e in trace_events:
        if e.get("event") != "grounding_checked":
            continue
        n_claims += e.get("n_claims", 0)
        n_supported += e.get("n_supported", 0)
        fabricated.extend(e.get("fabricated_citations", []))
    return G4Result(n_claims, n_supported, fabricated)
