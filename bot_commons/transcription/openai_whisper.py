"""Přepis přes OpenAI Whisper API (``/v1/audio/transcriptions``).

Logika je beze změny přenesená z původního ``bot_commons.whisper`` – včetně
retry na HTTP 429 podle hlavičky ``retry-after``.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from bot_commons.transcription.base import (
    TranscriptionError,
    TranscriptionResult,
    guess_mime_type,
    retry_delay,
)

log = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-1"
_TIMEOUT = 60.0
_MAX_RETRIES = 3


class OpenAIWhisperProvider:
    """Provider nad hostovaným OpenAI Whisper API."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = _TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if not api_key:
            raise TranscriptionError(
                self.name, "chybí API klíč (nastav OPENAI_API_KEY nebo předej api_key)"
            )
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "voice.ogg",
        language: str = "cs",
    ) -> TranscriptionResult:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        mime = guess_mime_type(filename)
        for attempt in range(self._max_retries):
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                files = {"file": (filename, audio, mime)}
                data = {"model": self._model, "language": language}
                try:
                    r = await client.post(OPENAI_URL, files=files, data=data, headers=headers)
                except httpx.HTTPError as exc:
                    log.error("OpenAI Whisper selhalo: %s", exc)
                    raise TranscriptionError(self.name, f"HTTP chyba: {exc}") from exc

                if r.status_code == 429:
                    wait = retry_delay(r, attempt)
                    log.warning(
                        "OpenAI rate limit, čekám %ss (pokus %d/%d)",
                        wait,
                        attempt + 1,
                        self._max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue

                if r.status_code >= 400:
                    log.error("OpenAI Whisper vrátilo %d: %s", r.status_code, r.text[:500])
                    raise TranscriptionError(
                        self.name, f"API vrátilo {r.status_code}: {r.text[:200]}"
                    )

                text: str = r.json()["text"].strip()
                log.info(
                    "transcribe provider=openai model=%s chars=%d", self._model, len(text)
                )
                return TranscriptionResult(
                    text=text,
                    provider=self.name,
                    model=self._model,
                    language=language,
                )

        raise TranscriptionError(
            self.name, f"API vrátilo 429 po {self._max_retries} pokusech"
        )
