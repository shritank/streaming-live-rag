"""Cost model: price-per-token comes from config, never hardcoded in code."""
from __future__ import annotations


def call_cost_usd(model: str, tokens_in: int, tokens_out: int,
                   price_table: dict[str, tuple[float, float]]) -> float:
    price_in, price_out = price_table.get(model, (0.0, 0.0))
    return tokens_in / 1000 * price_in + tokens_out / 1000 * price_out


def total_cost_usd(events: list[dict]) -> float:
    return sum(e.get("cost_usd", 0.0) for e in events if e.get("event") == "llm_call")
