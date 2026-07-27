"""Testy vrstvy přepisu: providery, normalizace odpovědi, factory a env přepínání."""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from bot_commons.transcription import (
    GeminiTranscriptionProvider,
    LocalWhisperProvider,
    OpenAIWhisperProvider,
    TranscriptionError,
    TranscriptionResult,
    build_provider,
    provider_from_env,
)
from bot_commons.transcription import gemini as gemini_mod
from bot_commons.transcription.openai_whisper import OPENAI_URL

GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-3.6-flash:generateContent"
)
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"

_ENV_VARS = (
    "TRANSCRIPTION_PROVIDER",
    "OPENAI_API_KEY",
    "WHISPER_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "WHISPER_LOCAL_URL",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Izoluje testy od skutečného prostředí vývojáře."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _gemini_ok(text: str) -> httpx.Response:
    return httpx.Response(
        200, json={"candidates": [{"content": {"parts": [{"text": text}]}}]}
    )


# --- společný tvar výsledku napříč providery ---------------------------------


@respx.mock
async def test_openai_result_shape():
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"text": " ahoj "}))
    result = await OpenAIWhisperProvider("sk-test").transcribe(b"audio")
    assert isinstance(result, TranscriptionResult)
    assert (result.text, result.provider, result.model, result.language) == (
        "ahoj",
        "openai",
        "whisper-1",
        "cs",
    )


@respx.mock
async def test_gemini_result_shape():
    respx.post(GENERATE_URL).mock(return_value=_gemini_ok(" ahoj "))
    result = await GeminiTranscriptionProvider("g-test").transcribe(b"audio")
    assert isinstance(result, TranscriptionResult)
    assert (result.text, result.provider, result.model, result.language) == (
        "ahoj",
        "gemini",
        "gemini-3.6-flash",
        "cs",
    )


@respx.mock
async def test_local_result_shape():
    respx.post("http://whisper:9000/transcribe").mock(
        return_value=httpx.Response(200, json={"text": " ahoj "})
    )
    result = await LocalWhisperProvider("http://whisper:9000/").transcribe(b"audio")
    assert (result.text, result.provider, result.model) == ("ahoj", "local", None)


# --- Gemini: tvar požadavku --------------------------------------------------


@respx.mock
async def test_gemini_sends_inline_audio_and_czech_prompt():
    route = respx.post(GENERATE_URL).mock(return_value=_gemini_ok("text"))
    await GeminiTranscriptionProvider("g-test").transcribe(b"raw-audio", filename="voice.ogg")

    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == "g-test"

    parts = json.loads(request.content)["contents"][0]["parts"]
    assert "Czech" in parts[0]["text"]
    assert parts[1]["inline_data"]["mime_type"] == "audio/ogg"
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"raw-audio"


@respx.mock
async def test_gemini_mime_type_follows_filename():
    route = respx.post(GENERATE_URL).mock(return_value=_gemini_ok("text"))
    await GeminiTranscriptionProvider("g-test").transcribe(b"audio", filename="note.mp3")

    parts = json.loads(route.calls.last.request.content)["contents"][0]["parts"]
    assert parts[1]["inline_data"]["mime_type"] == "audio/mp3"


@respx.mock
async def test_gemini_language_hint_is_configurable():
    route = respx.post(GENERATE_URL).mock(return_value=_gemini_ok("text"))
    await GeminiTranscriptionProvider("g-test").transcribe(b"audio", language="en")

    parts = json.loads(route.calls.last.request.content)["contents"][0]["parts"]
    assert "English" in parts[0]["text"]


@respx.mock
async def test_gemini_custom_model_hits_its_own_url():
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.5-flash:generateContent"
    ).mock(return_value=_gemini_ok("text"))
    await GeminiTranscriptionProvider("g-test", model="gemini-3.5-flash").transcribe(b"audio")
    assert route.called


# --- Gemini: normalizace odpovědi -------------------------------------------


@respx.mock
async def test_gemini_joins_parts_and_skips_thoughts():
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "přemýšlím", "thought": True},
                                {"text": "ahoj "},
                                {"text": "světe"},
                            ]
                        }
                    }
                ]
            },
        )
    )
    result = await GeminiTranscriptionProvider("g-test").transcribe(b"audio")
    assert result.text == "ahoj světe"


@respx.mock
async def test_gemini_empty_speech_returns_empty_string():
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(
            200, json={"candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]}
        )
    )
    result = await GeminiTranscriptionProvider("g-test").transcribe(b"audio")
    assert result.text == ""


@respx.mock
async def test_gemini_blocked_prompt_raises():
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})
    )
    with pytest.raises(TranscriptionError, match="SAFETY"):
        await GeminiTranscriptionProvider("g-test").transcribe(b"audio")


@respx.mock
async def test_gemini_no_candidates_raises():
    respx.post(GENERATE_URL).mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(TranscriptionError, match="candidates"):
        await GeminiTranscriptionProvider("g-test").transcribe(b"audio")


@respx.mock
async def test_gemini_bad_finish_reason_raises():
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(
            200, json={"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
        )
    )
    with pytest.raises(TranscriptionError, match="MAX_TOKENS"):
        await GeminiTranscriptionProvider("g-test").transcribe(b"audio")


# --- Gemini: chyby a retry ---------------------------------------------------


@respx.mock
async def test_gemini_http_error_raises_with_provider_name():
    respx.post(GENERATE_URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(TranscriptionError, match=r"\[gemini\]"):
        await GeminiTranscriptionProvider("g-test").transcribe(b"audio")


@respx.mock
async def test_gemini_timeout_raises_with_provider_name():
    respx.post(GENERATE_URL).mock(side_effect=httpx.ReadTimeout("timeout"))
    with pytest.raises(TranscriptionError, match=r"\[gemini\]"):
        await GeminiTranscriptionProvider("g-test").transcribe(b"audio")


@respx.mock
async def test_gemini_retries_on_429(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(gemini_mod.asyncio, "sleep", _no_sleep)
    respx.post(GENERATE_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}),
            _gemini_ok("ok"),
        ]
    )
    result = await GeminiTranscriptionProvider("g-test").transcribe(b"audio")
    assert result.text == "ok"


@respx.mock
async def test_gemini_gives_up_after_retries(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(gemini_mod.asyncio, "sleep", _no_sleep)
    respx.post(GENERATE_URL).mock(
        return_value=httpx.Response(429, headers={"retry-after": "0"})
    )
    with pytest.raises(TranscriptionError, match="429"):
        await GeminiTranscriptionProvider("g-test").transcribe(b"audio")


async def test_gemini_missing_key_raises_before_any_call():
    with pytest.raises(TranscriptionError, match="GEMINI_API_KEY"):
        GeminiTranscriptionProvider("")


async def test_openai_missing_key_raises_before_any_call():
    with pytest.raises(TranscriptionError, match="OPENAI_API_KEY"):
        OpenAIWhisperProvider("")


# --- Gemini: Files API pro velké audio ---------------------------------------


@respx.mock
async def test_gemini_large_audio_goes_through_files_api():
    respx.post(UPLOAD_URL).mock(
        return_value=httpx.Response(
            200, headers={"x-goog-upload-url": "https://upload.example/session"}
        )
    )
    respx.post("https://upload.example/session").mock(
        return_value=httpx.Response(
            200,
            json={
                "file": {
                    "name": "files/abc",
                    "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                    "state": "ACTIVE",
                }
            },
        )
    )
    generate = respx.post(GENERATE_URL).mock(return_value=_gemini_ok("velký přepis"))

    provider = GeminiTranscriptionProvider("g-test", inline_limit_bytes=4)
    result = await provider.transcribe(b"much-larger-audio")

    assert result.text == "velký přepis"
    parts = json.loads(generate.calls.last.request.content)["contents"][0]["parts"]
    assert parts[1]["file_data"]["file_uri"].endswith("/files/abc")
    assert "inline_data" not in parts[1]


@respx.mock
async def test_gemini_waits_for_file_processing(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(gemini_mod.asyncio, "sleep", _no_sleep)
    respx.post(UPLOAD_URL).mock(
        return_value=httpx.Response(
            200, headers={"x-goog-upload-url": "https://upload.example/session"}
        )
    )
    respx.post("https://upload.example/session").mock(
        return_value=httpx.Response(
            200,
            json={
                "file": {
                    "name": "files/abc",
                    "uri": "https://example/files/abc",
                    "state": "PROCESSING",
                }
            },
        )
    )
    status = respx.get(
        "https://generativelanguage.googleapis.com/v1beta/files/abc"
    ).mock(
        side_effect=[
            httpx.Response(200, json={"state": "PROCESSING"}),
            httpx.Response(200, json={"state": "ACTIVE"}),
        ]
    )
    respx.post(GENERATE_URL).mock(return_value=_gemini_ok("hotovo"))

    provider = GeminiTranscriptionProvider("g-test", inline_limit_bytes=4)
    result = await provider.transcribe(b"much-larger-audio")

    assert result.text == "hotovo"
    assert status.call_count == 2


@respx.mock
async def test_gemini_failed_upload_raises():
    respx.post(UPLOAD_URL).mock(return_value=httpx.Response(403, text="nope"))
    provider = GeminiTranscriptionProvider("g-test", inline_limit_bytes=4)
    with pytest.raises(TranscriptionError, match="Files API"):
        await provider.transcribe(b"much-larger-audio")


# --- factory a env přepínání -------------------------------------------------


def test_build_provider_returns_right_types():
    assert isinstance(build_provider("openai", api_key="k"), OpenAIWhisperProvider)
    assert isinstance(build_provider("gemini", api_key="k"), GeminiTranscriptionProvider)
    assert isinstance(
        build_provider("local", local_url="http://x:9000"), LocalWhisperProvider
    )


def test_build_provider_rejects_unknown_name():
    with pytest.raises(ValueError, match="nonsense"):
        build_provider("nonsense")


def test_provider_from_env_defaults_to_openai(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    assert isinstance(provider_from_env(), OpenAIWhisperProvider)


def test_provider_from_env_switches_to_gemini(clean_env):
    clean_env.setenv("TRANSCRIPTION_PROVIDER", "gemini")
    clean_env.setenv("GEMINI_API_KEY", "g-test")
    assert isinstance(provider_from_env(), GeminiTranscriptionProvider)


def test_provider_from_env_is_case_insensitive(clean_env):
    clean_env.setenv("TRANSCRIPTION_PROVIDER", "  GEMINI ")
    clean_env.setenv("GEMINI_API_KEY", "g-test")
    assert isinstance(provider_from_env(), GeminiTranscriptionProvider)


@respx.mock
async def test_provider_from_env_reads_model_override(clean_env):
    clean_env.setenv("TRANSCRIPTION_PROVIDER", "gemini")
    clean_env.setenv("GEMINI_API_KEY", "g-test")
    clean_env.setenv("GEMINI_MODEL", "gemini-3.5-flash")
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.5-flash:generateContent"
    ).mock(return_value=_gemini_ok("text"))

    await provider_from_env().transcribe(b"audio")
    assert route.called


def test_provider_from_env_local(clean_env):
    clean_env.setenv("TRANSCRIPTION_PROVIDER", "local")
    clean_env.setenv("WHISPER_LOCAL_URL", "http://whisper:9000")
    assert isinstance(provider_from_env(), LocalWhisperProvider)


def test_provider_from_env_needs_only_the_chosen_key(clean_env):
    """Klíč nepoužitého providera nastavený být nemusí."""
    clean_env.setenv("TRANSCRIPTION_PROVIDER", "gemini")
    clean_env.setenv("GEMINI_API_KEY", "g-test")
    assert isinstance(provider_from_env(), GeminiTranscriptionProvider)


def test_provider_from_env_fails_loudly_on_missing_key(clean_env):
    """Chybějící klíč = hlasitá chyba, žádné tiché přepnutí na druhého providera."""
    clean_env.setenv("TRANSCRIPTION_PROVIDER", "gemini")
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(TranscriptionError, match="GEMINI_API_KEY"):
        provider_from_env()


@respx.mock
async def test_gemini_failure_never_falls_back_to_openai():
    """Selhání zvoleného providera nesmí potichu zavolat toho druhého."""
    respx.post(GENERATE_URL).mock(return_value=httpx.Response(500, text="boom"))
    openai_route = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json={"text": "tohle se nesmí použít"})
    )
    with pytest.raises(TranscriptionError):
        await GeminiTranscriptionProvider("g-test").transcribe(b"audio")
    assert not openai_route.called
