"""Vrstva přepisu hlasu – jedno rozhraní, tři implementace.

```python
from bot_commons.transcription import provider_from_env, build_provider

provider = provider_from_env()                      # podle TRANSCRIPTION_PROVIDER
provider = build_provider("gemini", api_key=KEY)    # explicitně, i per request

result = await provider.transcribe(audio_bytes)
result.text, result.provider, result.model
```
"""

from __future__ import annotations

from bot_commons.transcription.base import (
    TranscriptionError,
    TranscriptionProvider,
    TranscriptionResult,
)
from bot_commons.transcription.factory import (
    DEFAULT_PROVIDER,
    PROVIDERS,
    build_provider,
    provider_from_env,
)
from bot_commons.transcription.gemini import GeminiTranscriptionProvider
from bot_commons.transcription.local_whisper import LocalWhisperProvider
from bot_commons.transcription.openai_whisper import OpenAIWhisperProvider

__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "GeminiTranscriptionProvider",
    "LocalWhisperProvider",
    "OpenAIWhisperProvider",
    "TranscriptionError",
    "TranscriptionProvider",
    "TranscriptionResult",
    "build_provider",
    "provider_from_env",
]
