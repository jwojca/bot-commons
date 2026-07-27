"""Přepis přes vlastní běžící whisper službu (``POST {base_url}/transcribe``).

Logika beze změny přenesená z původního ``bot_commons.whisper``.
"""

from __future__ import annotations

import logging

import httpx

from bot_commons.transcription.base import (
    TranscriptionError,
    TranscriptionResult,
    guess_mime_type,
)

log = logging.getLogger(__name__)

_TIMEOUT = 180.0


class LocalWhisperProvider:
    """Provider nad self-hosted whisper službou. Nepotřebuje API klíč."""

    name = "local"

    def __init__(self, local_url: str, *, timeout: float = _TIMEOUT) -> None:
        if not local_url:
            raise ValueError("provider='local' vyžaduje local_url")
        self._local_url = local_url.rstrip("/")
        self._timeout = timeout

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "voice.ogg",
        language: str = "cs",
    ) -> TranscriptionResult:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            files = {"audio": (filename, audio, guess_mime_type(filename))}
            try:
                r = await client.post(f"{self._local_url}/transcribe", files=files)
            except httpx.HTTPError as exc:
                log.error("Lokální whisper selhal: %s", exc)
                raise TranscriptionError(self.name, f"HTTP chyba: {exc}") from exc

            if r.status_code >= 400:
                log.error("Lokální whisper vrátil %d: %s", r.status_code, r.text[:500])
                raise TranscriptionError(
                    self.name, f"služba vrátila {r.status_code}: {r.text[:200]}"
                )

            text: str = r.json()["text"]
            log.info("transcribe provider=local chars=%d", len(text.strip()))
            return TranscriptionResult(
                text=text.strip(),
                provider=self.name,
                model=None,
                language=language,
            )
