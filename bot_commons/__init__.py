"""bot-commons – sdílená logika napříč Telegram boty.

Veřejné API:
- :mod:`bot_commons.transcription` – přepis hlasu (OpenAI Whisper / Gemini / lokální).
- :mod:`bot_commons.whisper` – zpětně kompatibilní ``transcribe()`` fasáda.
- :mod:`bot_commons.pricing` – výpočet a formátování ceny za tokeny.
- :mod:`bot_commons.jsonparse` – parsování JSON z Claude odpovědí.
- :mod:`bot_commons.config` – lehké env/logging helpery.
"""

from __future__ import annotations

from bot_commons.config import (
    ConfigError,
    env_flag,
    get_env,
    require_env,
    setup_logging,
)
from bot_commons.jsonparse import parse_json_object, parse_json_objects
from bot_commons.pricing import (
    PRICING,
    USD_TO_CZK,
    add_usage,
    format_usage,
    record_usage,
)
from bot_commons.transcription import (
    GeminiTranscriptionProvider,
    LocalWhisperProvider,
    OpenAIWhisperProvider,
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionResult,
    build_provider,
    provider_from_env,
)
from bot_commons.whisper import transcribe

__version__ = "0.2.1"

__all__ = [
    "PRICING",
    "USD_TO_CZK",
    "ConfigError",
    "GeminiTranscriptionProvider",
    "LocalWhisperProvider",
    "OpenAIWhisperProvider",
    "TranscriptionError",
    "TranscriptionProvider",
    "TranscriptionResult",
    "add_usage",
    "build_provider",
    "env_flag",
    "format_usage",
    "get_env",
    "parse_json_object",
    "parse_json_objects",
    "provider_from_env",
    "record_usage",
    "require_env",
    "setup_logging",
    "transcribe",
]
