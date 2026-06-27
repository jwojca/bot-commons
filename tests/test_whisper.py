"""Testy whisper přepisu (httpx přes respx mock)."""

from __future__ import annotations

import httpx
import pytest
import respx

from bot_commons import whisper


@respx.mock
async def test_transcribe_openai_ok():
    route = respx.post(whisper.OPENAI_URL).mock(
        return_value=httpx.Response(200, json={"text": "  ahoj světe  "})
    )
    out = await whisper.transcribe(b"audio", api_key="sk-test")
    assert out == "ahoj světe"
    assert route.called
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"


@respx.mock
async def test_transcribe_openai_retries_on_429(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(whisper.asyncio, "sleep", _no_sleep)
    respx.post(whisper.OPENAI_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, json={"text": "ok"}),
        ]
    )
    out = await whisper.transcribe(b"audio", api_key="sk-test")
    assert out == "ok"


@respx.mock
async def test_transcribe_openai_gives_up_after_retries(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(whisper.asyncio, "sleep", _no_sleep)
    respx.post(whisper.OPENAI_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": "0"})
    )
    with pytest.raises(RuntimeError):
        await whisper.transcribe(b"audio", api_key="sk-test")


@respx.mock
async def test_transcribe_local():
    route = respx.post("http://whisper:9000/transcribe").mock(
        return_value=httpx.Response(200, json={"text": "lokální přepis"})
    )
    out = await whisper.transcribe(
        b"audio", provider="local", local_url="http://whisper:9000"
    )
    assert out == "lokální přepis"
    assert route.called


async def test_local_requires_url():
    with pytest.raises(ValueError):
        await whisper.transcribe(b"audio", provider="local")


async def test_unknown_provider():
    with pytest.raises(ValueError):
        await whisper.transcribe(b"audio", provider="nonsense")
