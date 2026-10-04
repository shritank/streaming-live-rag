"""provider='local': OpenAI-compatible HTTP client for a locally hosted open model (httpx, no network in tests)."""
from __future__ import annotations

import json

import httpx
import pytest

from streaming_rag.config import load_config
from streaming_rag.llm import LocalLLM, build_llm_client


def _cfg():
    cfg = load_config()
    cfg.llm.provider, cfg.llm.model, cfg.llm.base_url, cfg.llm.max_retries = "local", "qwen-test", "http://llm.test/v1", 1
    return cfg


class _Events:
    def __init__(self):
        self.items = []

    def emit(self, event, **fields):
        self.items.append((event, fields))


async def test_local_provider_posts_chat_completions_and_returns_the_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": true}"}}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 3}})

    events = _Events()
    llm = LocalLLM(_cfg(), events, transport=httpx.MockTransport(handler))
    out = await llm.complete(purpose="t", system="sys", prompt="hi", json_mode=True, max_tokens=64)
    assert out.text == '{"ok": true}' and (out.tokens_in, out.tokens_out) == (11, 3)
    assert seen["url"] == "http://llm.test/v1/chat/completions"
    assert seen["body"]["model"] == "qwen-test" and seen["body"]["max_tokens"] == 64
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]
    assert events.items[0][0] == "llm_call" and events.items[0][1]["cost_usd"] == 0.0


async def test_local_provider_retries_once_then_raises_and_reports_the_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={})

    events = _Events()
    llm = LocalLLM(_cfg(), events, transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await llm.complete(purpose="t", system="s", prompt="p")
    assert calls["n"] == 2                                          # first try + one retry
    assert events.items[-1][0] == "error" and events.items[-1][1]["where"] == "llm.local"


def test_build_llm_client_selects_the_local_provider():
    assert isinstance(build_llm_client(_cfg()), LocalLLM)
