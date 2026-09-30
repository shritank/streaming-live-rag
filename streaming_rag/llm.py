"""Async LLMClient implementations.

The engine only ever depends on the `LLMClient` Protocol from contracts.py.
`build_llm_client(config, telemetry)` selects an implementation from
`config.llm.provider`:

  - "fake"   : deterministic, offline, zero-cost. Default. Used by mocks,
               unit tests and any environment without an API key.
  - "openai" : AsyncOpenAI-backed, only imported if selected.
  - "gemini" : google-genai async client, only imported if selected.

All real providers are wrapped with: temperature=0, a fixed seed where the
provider supports it, a timeout, one retry on transient errors, and a
single `llm_call` telemetry event per call carrying tokens/cost/latency.
Never call a sync client directly on the event loop — providers without a
native async client must be wrapped in `asyncio.to_thread`.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time

from .config import Config
from .contracts import LLMResponse, Telemetry, NullTelemetry


def _price_for(model: str, price_table: dict[str, tuple[float, float]]) -> tuple[float, float]:
    return price_table.get(model, (0.0, 0.0))


def _estimate_tokens(text: str) -> int:
    # Cheap, deterministic token estimate (no tokenizer dependency): ~4 chars/token.
    return max(1, len(text) // 4)


class FakeLLM:
    """Deterministic offline LLM. Never makes network calls.

    Produces stable output for a given (purpose, system, prompt) so tests
    and local replay are fully reproducible without a provider key.
    """

    def __init__(self, config: Config, telemetry: Telemetry | None = None):
        self._config = config
        self._telemetry = telemetry or NullTelemetry()
        self._model = "fake-model"

    async def complete(self, *, purpose: str, system: str, prompt: str,
                        json_mode: bool = False, max_tokens: int = 512) -> LLMResponse:
        start = time.perf_counter()
        # Deterministic "generation": a stable digest-derived echo, never invents facts.
        digest = hashlib.sha256(f"{purpose}|{system}|{prompt}".encode()).hexdigest()[:8]
        text = prompt if not json_mode else "{}"
        await asyncio.sleep(0)  # yield control; never block the loop
        latency_ms = (time.perf_counter() - start) * 1000
        tokens_in = _estimate_tokens(system + prompt)
        tokens_out = _estimate_tokens(text)
        price_in, price_out = _price_for(self._model, self._config.llm.price_table)
        cost = tokens_in / 1000 * price_in + tokens_out / 1000 * price_out
        self._telemetry.emit(
            "llm_call",
            purpose=purpose,
            model=self._model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
            digest=digest,
        )
        return LLMResponse(text=text, model=self._model, tokens_in=tokens_in,
                            tokens_out=tokens_out, latency_ms=latency_ms)


class OpenAILLM:
    """Async OpenAI-backed LLMClient. Only constructed when provider=='openai'."""

    def __init__(self, config: Config, telemetry: Telemetry | None = None):
        try:
            from openai import AsyncOpenAI  # imported lazily; optional dependency
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER=openai requires the 'openai' package; "
                "add it to requirements.lock and pip install it"
            ) from e
        api_key = os.environ.get(config.llm.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key in env var {config.llm.api_key_env}")
        self._client = AsyncOpenAI(api_key=api_key, timeout=config.llm.timeout_s)
        self._config = config
        self._telemetry = telemetry or NullTelemetry()

    async def complete(self, *, purpose: str, system: str, prompt: str,
                        json_mode: bool = False, max_tokens: int = 512) -> LLMResponse:
        model = self._config.llm.model
        attempts = self._config.llm.max_retries + 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    temperature=self._config.llm.temperature,
                    seed=self._config.llm.seed,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} if json_mode else None,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                )
                latency_ms = (time.perf_counter() - start) * 1000
                text = resp.choices[0].message.content or ""
                tokens_in = resp.usage.prompt_tokens if resp.usage else _estimate_tokens(system + prompt)
                tokens_out = resp.usage.completion_tokens if resp.usage else _estimate_tokens(text)
                price_in, price_out = _price_for(model, self._config.llm.price_table)
                cost = tokens_in / 1000 * price_in + tokens_out / 1000 * price_out
                self._telemetry.emit(
                    "llm_call", purpose=purpose, model=model, tokens_in=tokens_in,
                    tokens_out=tokens_out, latency_ms=latency_ms, cost_usd=cost,
                )
                return LLMResponse(text=text, model=model, tokens_in=tokens_in,
                                    tokens_out=tokens_out, latency_ms=latency_ms)
            except Exception as e:  # transient error: retry once, then raise
                last_err = e
                if attempt < attempts - 1:
                    await asyncio.sleep(0.1)
                    continue
                self._telemetry.emit("error", where="llm.openai", error_type=type(e).__name__, message=str(e))
                raise
        assert last_err is not None
        raise last_err


class GeminiLLM:
    """Async google-genai-backed LLMClient. Only constructed when provider=='gemini'."""

    def __init__(self, config: Config, telemetry: Telemetry | None = None):
        try:
            from google import genai  # imported lazily; optional dependency
        except ImportError as e:
            raise RuntimeError(
                "LLM_PROVIDER=gemini requires the 'google-genai' package; "
                "add it to requirements.lock and pip install it"
            ) from e
        api_key = os.environ.get(config.llm.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key in env var {config.llm.api_key_env}")
        self._client = genai.Client(api_key=api_key)
        self._config = config
        self._telemetry = telemetry or NullTelemetry()

    async def complete(self, *, purpose: str, system: str, prompt: str,
                        json_mode: bool = False, max_tokens: int = 512) -> LLMResponse:
        model = self._config.llm.model
        attempts = self._config.llm.max_retries + 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                resp = await self._client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "system_instruction": system,
                        "temperature": self._config.llm.temperature,
                        "seed": self._config.llm.seed,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json" if json_mode else "text/plain",
                    },
                )
                latency_ms = (time.perf_counter() - start) * 1000
                text = resp.text or ""
                usage = getattr(resp, "usage_metadata", None)
                tokens_in = getattr(usage, "prompt_token_count", None) or _estimate_tokens(system + prompt)
                tokens_out = getattr(usage, "candidates_token_count", None) or _estimate_tokens(text)
                price_in, price_out = _price_for(model, self._config.llm.price_table)
                cost = tokens_in / 1000 * price_in + tokens_out / 1000 * price_out
                self._telemetry.emit(
                    "llm_call", purpose=purpose, model=model, tokens_in=tokens_in,
                    tokens_out=tokens_out, latency_ms=latency_ms, cost_usd=cost,
                )
                return LLMResponse(text=text, model=model, tokens_in=tokens_in,
                                    tokens_out=tokens_out, latency_ms=latency_ms)
            except Exception as e:
                last_err = e
                if attempt < attempts - 1:
                    await asyncio.sleep(0.1)
                    continue
                self._telemetry.emit("error", where="llm.gemini", error_type=type(e).__name__, message=str(e))
                raise
        assert last_err is not None
        raise last_err


def build_llm_client(config: Config, telemetry: Telemetry | None = None):
    provider = config.llm.provider
    if provider == "fake":
        return FakeLLM(config, telemetry)
    if provider == "openai":
        return OpenAILLM(config, telemetry)
    if provider == "gemini":
        return GeminiLLM(config, telemetry)
    raise ValueError(f"unknown LLM_PROVIDER: {provider!r}")
