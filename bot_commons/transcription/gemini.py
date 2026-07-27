"""Přepis přes Gemini API – multimodální audio vstup, ne Cloud Speech-to-Text.

Malé audio jde inline jako base64 (``inline_data``), velké se nejdřív nahraje
přes Files API a pošle se jen odkaz (``file_data``). Hranice je
:data:`INLINE_LIMIT_BYTES`; Telegram hlasovky se do inline vejdou vždycky.

Voláme raw REST přes ``httpx`` (stejně jako OpenAI varianta), aby si knihovna
nepřitáhla další závislost a šla mockovat přes ``respx``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

import httpx

from bot_commons.transcription.base import (
    TranscriptionError,
    TranscriptionResult,
    guess_mime_type,
    language_name,
    retry_delay,
)

log = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
DEFAULT_MODEL = "gemini-3.6-flash"

# Limit celého requestu je 20 MB a base64 nafoukne data o ~33 %.
# 12 MiB syrového audia → ~16 MB v JSONu, což je bezpečná rezerva.
INLINE_LIMIT_BYTES = 12 * 1024 * 1024

_TIMEOUT = 120.0
_MAX_RETRIES = 3
_FILE_POLL_INTERVAL = 1.0
_FILE_POLL_ATTEMPTS = 30


def build_prompt(language: str) -> str:
    """Prompt pro doslovný přepis s jazykovým hintem."""
    return (
        f"Transcribe this audio verbatim in {language_name(language)}. "
        "Return only the transcript text – no commentary, no timestamps, "
        "no speaker labels, no markdown formatting. "
        "If there is no intelligible speech, return an empty string."
    )


class GeminiTranscriptionProvider:
    """Provider nad Gemini ``generateContent`` s audio vstupem."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        timeout: float = _TIMEOUT,
        max_retries: int = _MAX_RETRIES,
        inline_limit_bytes: int = INLINE_LIMIT_BYTES,
    ) -> None:
        if not api_key:
            raise TranscriptionError(
                self.name, "chybí API klíč (nastav GEMINI_API_KEY nebo předej api_key)"
            )
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._inline_limit = inline_limit_bytes

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "voice.ogg",
        language: str = "cs",
    ) -> TranscriptionResult:
        mime = guess_mime_type(filename)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            if len(audio) > self._inline_limit:
                log.info(
                    "gemini: %d B > inline limit, nahrávám přes Files API", len(audio)
                )
                file_uri = await self._upload_file(client, audio, filename, mime)
                audio_part: dict[str, Any] = {
                    "file_data": {"mime_type": mime, "file_uri": file_uri}
                }
            else:
                audio_part = {
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(audio).decode("ascii"),
                    }
                }

            payload = {
                "contents": [
                    {"parts": [{"text": build_prompt(language)}, audio_part]}
                ],
                "generationConfig": {"temperature": 0.0},
            }
            data = await self._generate(client, payload)

        text = _extract_text(data, self.name)
        log.info("transcribe provider=gemini model=%s chars=%d", self._model, len(text))
        return TranscriptionResult(
            text=text,
            provider=self.name,
            model=self._model,
            language=language,
        )

    async def _generate(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> dict[str, Any]:
        url = f"{API_BASE}/models/{self._model}:generateContent"
        for attempt in range(self._max_retries):
            try:
                r = await client.post(url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                log.error("Gemini API selhalo: %s", exc)
                raise TranscriptionError(self.name, f"HTTP chyba: {exc}") from exc

            if r.status_code == 429:
                wait = retry_delay(r, attempt)
                log.warning(
                    "Gemini rate limit, čekám %ss (pokus %d/%d)",
                    wait,
                    attempt + 1,
                    self._max_retries,
                )
                await asyncio.sleep(wait)
                continue

            if r.status_code >= 400:
                log.error("Gemini API vrátilo %d: %s", r.status_code, r.text[:500])
                raise TranscriptionError(
                    self.name, f"API vrátilo {r.status_code}: {r.text[:200]}"
                )

            result: dict[str, Any] = r.json()
            return result

        raise TranscriptionError(
            self.name, f"API vrátilo 429 po {self._max_retries} pokusech"
        )

    async def _upload_file(
        self, client: httpx.AsyncClient, audio: bytes, filename: str, mime: str
    ) -> str:
        """Resumable upload přes Files API; vrátí ``file_uri`` připravený k použití."""
        start_headers = {
            **self._headers,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(audio)),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        }
        try:
            start = await client.post(
                UPLOAD_URL,
                json={"file": {"display_name": filename}},
                headers=start_headers,
            )
        except httpx.HTTPError as exc:
            raise TranscriptionError(self.name, f"Files API start selhal: {exc}") from exc

        if start.status_code >= 400:
            raise TranscriptionError(
                self.name, f"Files API start vrátil {start.status_code}: {start.text[:200]}"
            )

        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise TranscriptionError(self.name, "Files API nevrátilo x-goog-upload-url")

        try:
            done = await client.post(
                upload_url,
                content=audio,
                headers={
                    "Content-Length": str(len(audio)),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
            )
        except httpx.HTTPError as exc:
            raise TranscriptionError(self.name, f"Files API upload selhal: {exc}") from exc

        if done.status_code >= 400:
            raise TranscriptionError(
                self.name, f"Files API upload vrátil {done.status_code}: {done.text[:200]}"
            )

        file_info = done.json().get("file") or {}
        uri = file_info.get("uri")
        if not uri:
            raise TranscriptionError(self.name, "Files API nevrátilo uri nahraného souboru")

        await self._await_file_active(client, file_info)
        return str(uri)

    async def _await_file_active(
        self, client: httpx.AsyncClient, file_info: dict[str, Any]
    ) -> None:
        """Audio se po uploadu chvíli zpracovává – počká na stav ACTIVE."""
        state = file_info.get("state")
        name = file_info.get("name")
        if state == "ACTIVE" or not name:
            return

        for _ in range(_FILE_POLL_ATTEMPTS):
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise TranscriptionError(self.name, "Files API: zpracování souboru selhalo")
            await asyncio.sleep(_FILE_POLL_INTERVAL)
            r = await client.get(f"{API_BASE}/{name}", headers=self._headers)
            if r.status_code >= 400:
                raise TranscriptionError(
                    self.name, f"Files API stav vrátil {r.status_code}: {r.text[:200]}"
                )
            state = r.json().get("state")

        raise TranscriptionError(self.name, "Files API: soubor se nezpracoval včas")


def _extract_text(data: dict[str, Any], provider: str) -> str:
    """Vytáhne text z ``generateContent`` odpovědi a znormalizuje ho na ``str``."""
    block_reason = (data.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
        raise TranscriptionError(provider, f"požadavek zablokován: {block_reason}")

    candidates = data.get("candidates") or []
    if not candidates:
        raise TranscriptionError(provider, "odpověď neobsahuje žádné candidates")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    # U 2.5 modelů můžou přijít i "thought" party – ty do přepisu nepatří.
    chunks = [
        p["text"] for p in parts if isinstance(p.get("text"), str) and not p.get("thought")
    ]

    if not chunks:
        finish = candidate.get("finishReason")
        if finish and finish != "STOP":
            raise TranscriptionError(provider, f"generování skončilo s finishReason={finish}")
        # Prázdná odpověď je legitimní výsledek pro audio beze slov.
        return ""

    return "".join(chunks).strip()
