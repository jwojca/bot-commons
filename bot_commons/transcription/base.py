"""Společné rozhraní a datové typy pro přepis hlasu.

Jádro vrstvy přepisu je **čisté** – nečte žádné env proměnné ani globály.
Providery dostanou klíč, model a URL v konstruktoru. Env přepínání řeší
opt-in :func:`bot_commons.transcription.provider_from_env`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

_DEFAULT_MIME = "audio/ogg"
_MIME_BY_SUFFIX = {
    "ogg": "audio/ogg",
    "oga": "audio/ogg",
    "opus": "audio/ogg",
    "mp3": "audio/mp3",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "aac": "audio/aac",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "webm": "audio/webm",
}

# ISO kód → název jazyka do promptu (Gemini bere hint textem, ne parametrem).
_LANGUAGE_NAMES = {
    "cs": "Czech",
    "sk": "Slovak",
    "en": "English",
    "de": "German",
    "pl": "Polish",
}


@dataclass(frozen=True)
class TranscriptionResult:
    """Výsledek přepisu, stejný tvar bez ohledu na providera.

    Attributes:
        text: Přepsaný text, už otrimovaný.
        provider: Který provider ho vyrobil (``"openai"``/``"gemini"``/``"local"``).
        model: Použitý model, pokud ho provider má (lokální whisper nemá).
        language: ISO kód jazyka, se kterým se volalo.
    """

    text: str
    provider: str
    model: str | None = None
    language: str | None = None


class TranscriptionError(RuntimeError):
    """Přepis selhal. Dědí z ``RuntimeError`` kvůli zpětné kompatibilitě.

    Nese název providera, aby z logu bylo hned vidět, co selhalo. Knihovna
    **nikdy** nepřepne tiše na jiného providera – chyba vždy probublá nahoru.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Rozhraní poskytovatele přepisu.

    Vlastní implementaci lze podstrčit kamkoli, kde se čeká provider – stačí
    dodržet tenhle tvar, dědit se z ničeho nemusí.
    """

    name: str

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "voice.ogg",
        language: str = "cs",
    ) -> TranscriptionResult:
        """Přepíše audio bajty. Při selhání vyhodí :class:`TranscriptionError`."""
        ...


def guess_mime_type(filename: str) -> str:
    """Odhadne MIME typ z přípony; Telegram voice (OGG/Opus) je výchozí."""
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _MIME_BY_SUFFIX.get(suffix, _DEFAULT_MIME)


def language_name(language: str) -> str:
    """Název jazyka pro prompt (``"cs"`` → ``"Czech"``), jinak vrátí kód."""
    return _LANGUAGE_NAMES.get(language.lower(), language)


def retry_delay(response: httpx.Response, attempt: int) -> int:
    """Kolik sekund počkat po HTTP 429 – z hlavičky ``retry-after``, jinak backoff."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return 5 * (attempt + 1)
    try:
        return int(raw)
    except ValueError:
        return 5 * (attempt + 1)
