"""Sestavení providera – jediné místo v knihovně, které smí číst env proměnné.

Jádro (``base``, ``openai_whisper``, ``gemini``, ``local_whisper``) zůstává
čisté a bere všechno explicitně. Kdo chce přepínat přes prostředí, zavolá si
:func:`provider_from_env` vědomě.
"""

from __future__ import annotations

import logging

from bot_commons.config import get_env
from bot_commons.transcription.base import TranscriptionProvider
from bot_commons.transcription.gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from bot_commons.transcription.gemini import GeminiTranscriptionProvider
from bot_commons.transcription.local_whisper import LocalWhisperProvider
from bot_commons.transcription.openai_whisper import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from bot_commons.transcription.openai_whisper import OpenAIWhisperProvider

log = logging.getLogger(__name__)

PROVIDERS = ("openai", "gemini", "local")
DEFAULT_PROVIDER = "openai"

ENV_PROVIDER = "TRANSCRIPTION_PROVIDER"
ENV_OPENAI_KEY = "OPENAI_API_KEY"
ENV_OPENAI_MODEL = "WHISPER_MODEL"
ENV_GEMINI_KEY = "GEMINI_API_KEY"
ENV_GEMINI_MODEL = "GEMINI_MODEL"
ENV_LOCAL_URL = "WHISPER_LOCAL_URL"


def build_provider(
    name: str = DEFAULT_PROVIDER,
    *,
    api_key: str | None = None,
    model: str | None = None,
    local_url: str | None = None,
) -> TranscriptionProvider:
    """Sestaví providera podle jména. Nic nečte z prostředí.

    Args:
        name: ``"openai"``, ``"gemini"`` nebo ``"local"``.
        api_key: Klíč daného providera (lokální whisper ho nepotřebuje).
        model: Model; ``None`` = výchozí model providera.
        local_url: Base URL lokální whisper služby (jen ``name="local"``).

    Raises:
        ValueError: Neznámé jméno providera nebo chybějící ``local_url``.
        TranscriptionError: Chybí API klíč pro providera, který ho vyžaduje.
    """
    if name == "openai":
        return OpenAIWhisperProvider(api_key or "", model=model or OPENAI_DEFAULT_MODEL)
    if name == "gemini":
        return GeminiTranscriptionProvider(api_key or "", model=model or GEMINI_DEFAULT_MODEL)
    if name == "local":
        if not local_url:
            raise ValueError("provider='local' vyžaduje local_url")
        return LocalWhisperProvider(local_url)
    raise ValueError(
        f"Neznámý transcription provider: {name!r} (známé: {', '.join(PROVIDERS)})"
    )


def provider_from_env(default: str = DEFAULT_PROVIDER) -> TranscriptionProvider:
    """Sestaví providera podle env proměnných.

    Čte:
        - ``TRANSCRIPTION_PROVIDER`` – ``openai`` (výchozí) | ``gemini`` | ``local``
        - ``OPENAI_API_KEY``, ``WHISPER_MODEL`` – pro ``openai``
        - ``GEMINI_API_KEY``, ``GEMINI_MODEL`` – pro ``gemini``
        - ``WHISPER_LOCAL_URL`` – pro ``local``

    Sestaví se **jen zvolený** provider, takže klíč toho druhého být nastavený
    nemusí. Když zvolený provider nejde sestavit, letí chyba – knihovna
    nikdy nepřepne tiše na jiného.
    """
    name = (get_env(ENV_PROVIDER) or default).strip().lower()
    provider = build_provider(
        name,
        api_key=_api_key_for(name),
        model=_model_for(name),
        local_url=get_env(ENV_LOCAL_URL),
    )
    log.info("transcription provider=%s (z %s)", name, ENV_PROVIDER)
    return provider


def _api_key_for(name: str) -> str | None:
    if name == "openai":
        return get_env(ENV_OPENAI_KEY)
    if name == "gemini":
        return get_env(ENV_GEMINI_KEY)
    return None


def _model_for(name: str) -> str | None:
    if name == "openai":
        return get_env(ENV_OPENAI_MODEL)
    if name == "gemini":
        return get_env(ENV_GEMINI_MODEL)
    return None
