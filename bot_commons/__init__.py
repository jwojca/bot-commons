"""bot-commons – sdílená logika napříč Telegram boty.

Veřejné API:
- :mod:`bot_commons.whisper` – přepis hlasu (OpenAI / lokální whisper).
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
from bot_commons.whisper import transcribe

__version__ = "0.1.0"

__all__ = [
    "PRICING",
    "USD_TO_CZK",
    "ConfigError",
    "add_usage",
    "env_flag",
    "format_usage",
    "get_env",
    "parse_json_object",
    "parse_json_objects",
    "record_usage",
    "require_env",
    "setup_logging",
    "transcribe",
]
