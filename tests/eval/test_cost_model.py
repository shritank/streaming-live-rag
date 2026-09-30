"""Task 4 section 4.3, cost accuracy: the summed cost_usd must equal tokens x price table EXACTLY (the price
comes from config, never from code). The second half of that acceptance item - agreement with a real provider's
reported usage within 2 % on a 5-call live sample - needs an API key and cannot be tested offline."""
from __future__ import annotations

import pytest

from streaming_rag.config import load_config
from streaming_rag.llm import FakeLLM
from streaming_rag.telemetry.cost import call_cost_usd, total_cost_usd
from streaming_rag.telemetry.sinks import BufferedTelemetry


def test_call_cost_is_tokens_times_the_configured_price():
    table = {"m": (0.00015, 0.0006)}
    assert call_cost_usd("m", 1000, 500, table) == pytest.approx(0.00015 + 0.5 * 0.0006, rel=0, abs=1e-15)
    assert call_cost_usd("m", 0, 0, table) == 0.0
    assert call_cost_usd("unpriced-model", 10_000, 10_000, table) == 0.0     # unknown model: free, never a KeyError


async def test_every_llm_call_event_carries_exactly_tokens_times_price_and_the_total_is_their_sum():
    config = load_config()
    config.llm.price_table = {"fake-model": (0.5, 2.0)}            # non-zero, so the check has teeth
    telemetry = BufferedTelemetry()
    llm = FakeLLM(config, telemetry)
    for i in range(5):
        await llm.complete(purpose="test", system="be brief", prompt="question number %d? " % i * (i + 1))
    calls = [e for e in telemetry.events if e["event"] == "llm_call"]
    assert len(calls) == 5
    for e in calls:
        assert e["cost_usd"] == e["tokens_in"] / 1000 * 0.5 + e["tokens_out"] / 1000 * 2.0
        assert e["cost_usd"] > 0
    assert total_cost_usd(telemetry.events) == sum(e["cost_usd"] for e in calls)


async def test_the_default_offline_provider_is_free():
    telemetry = BufferedTelemetry()
    llm = FakeLLM(load_config(), telemetry)
    await llm.complete(purpose="test", system="s", prompt="p")
    assert total_cost_usd(telemetry.events) == 0.0
