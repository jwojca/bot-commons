"""Speech-to-text přes OpenAI Whisper API nebo lokální whisper službu.

Sjednocuje tři dřívější kopie z jednotlivých botů. Funkce bere všechny vstupy
explicitně (api_key, model, provider…) – nečte žádné globály ani konfiguraci,
takže si ji každý bot obalí ve svém stylu.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

log = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"
_OPENAI_TIMEOUT = 60.0
_LOCAL_TIMEOUT = 180.0
_MAX_RETRIES = 3


async def transcribe(
    audio: bytes,
    *,
    api_key: str = "",
    filename: str = "voice.ogg",
    language: str = "cs",
    model: str = "whisper-1",
    provider: str = "openai",
    local_url: str | None = None,
) -> str:
    """Přepíše audio bajty na text.

    Args:
        audio: Obsah hlasové zprávy (Telegram voice = OGG/Opus).
        api_key: OpenAI API klíč (jen pro ``provider="openai"``).
        filename: Název s příponou kvůli detekci formátu na straně API.
        language: ISO kód jazyka (např. ``"cs"``).
        model: Whisper model (jen pro OpenAI provider).
        provider: ``"openai"`` (default) nebo ``"local"``.
        local_url: Base URL lokální whisper služby (jen pro ``provider="local"``).
    """
    if provider == "openai":
        return await _transcribe_openai(audio, api_key, filename, language, model)
    if provider == "local":
        if not local_url:
            raise ValueError("provider='local' vyžaduje local_url")
        return await _transcribe_local(audio, filename, local_url)
    raise ValueError(f"Neznámý whisper provider: {provider!r}")


async def _transcribe_openai(
    audio: bytes, api_key: str, filename: str, language: str, model: str
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(_MAX_RETRIES):
        async with httpx.AsyncClient(timeout=_OPENAI_TIMEOUT) as client:
            files = {"file": (filename, audio, "audio/ogg")}
            data = {"model": model, "language": language}
            r = await client.post(OPENAI_URL, files=files, data=data, headers=headers)
            if r.status_code == 429:
                wait = int(r.headers.get("retry-after", 5 * (attempt + 1)))
                log.warning("OpenAI rate limit, čekám %ss (pokus %d/%d)", wait, attempt + 1, _MAX_RETRIES)
                await asyncio.sleep(wait)
                continue
            r.raise_for_status()
            text: str = r.json()["text"]
            log.info("whisper.transcribe chars=%d", len(text.strip()))
            return text.strip()
    raise RuntimeError(f"OpenAI Whisper API vrátilo 429 po {_MAX_RETRIES} pokusech")


async def _transcribe_local(audio: bytes, filename: str, local_url: str) -> str:
    async with httpx.AsyncClient(timeout=_LOCAL_TIMEOUT) as client:
        files = {"audio": (filename, audio, "audio/ogg")}
        r = await client.post(f"{local_url}/transcribe", files=files)
        r.raise_for_status()
        text: str = r.json()["text"]
        return text.strip()
