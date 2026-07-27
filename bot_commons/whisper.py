"""Zpětně kompatibilní fasáda nad :mod:`bot_commons.transcription`.

Vlastní logika se přestěhovala do providerů. Tenhle modul zůstává, aby boti,
kteří volají ``transcribe(...)``, nemuseli měnit ani řádek – vrací pořád
plain ``str``.

Nový kód ať sáhne rovnou po providerech, dostane i metadata (kdo/čím přepisoval):

```python
from bot_commons.transcription import provider_from_env

provider = provider_from_env()
result = await provider.transcribe(audio)
```
"""

from __future__ import annotations

import logging

from bot_commons.transcription.factory import build_provider
from bot_commons.transcription.openai_whisper import OPENAI_URL

log = logging.getLogger(__name__)

__all__ = ["OPENAI_URL", "transcribe"]


async def transcribe(
    audio: bytes,
    *,
    api_key: str = "",
    filename: str = "voice.ogg",
    language: str = "cs",
    model: str | None = None,
    provider: str = "openai",
    local_url: str | None = None,
) -> str:
    """Přepíše audio bajty na text.

    Args:
        audio: Obsah hlasové zprávy (Telegram voice = OGG/Opus).
        api_key: API klíč zvoleného providera (``local`` ho nepotřebuje).
        filename: Název s příponou kvůli detekci formátu na straně API.
        language: ISO kód jazyka (např. ``"cs"``).
        model: Model providera; ``None`` = jeho výchozí
            (``whisper-1`` / ``gemini-3.6-flash``).
        provider: ``"openai"`` (default), ``"gemini"`` nebo ``"local"``.
        local_url: Base URL lokální whisper služby (jen ``provider="local"``).

    Returns:
        Přepsaný text. Metadata o providerovi tahle fasáda zahazuje – kdo je
        chce, ať volá providera přímo.
    """
    impl = build_provider(provider, api_key=api_key, model=model, local_url=local_url)
    result = await impl.transcribe(audio, filename=filename, language=language)
    return result.text
