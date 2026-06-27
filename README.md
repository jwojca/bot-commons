# bot-commons

Sdílená logika napříč mými Telegram boty (`crm-bot`, `jidlo-bot`, `vylety-bot`).
Jedno místo pravdy pro věci, které se dřív kopírovaly a rozjížděly.

## Co obsahuje

| Modul | Co dělá |
|---|---|
| `bot_commons.whisper` | Přepis hlasu přes OpenAI Whisper API (429 retry) nebo lokální whisper službu. |
| `bot_commons.pricing` | Výpočet ceny za tokeny z Anthropic odpovědi (per-model breakdown, cache, CZK) + formátování. |
| `bot_commons.jsonparse` | Parsování JSON z Claude odpovědí (markdown fence, pole, víc objektů za sebou). |
| `bot_commons.config` | Lehké env/logging helpery (`require_env`, `env_flag`, `setup_logging`). |

Funkce berou vstupy **explicitně** (api_key, model, …) – nečtou žádné globály,
takže si je každý bot obalí ve svém stylu.

## Instalace

V botovi (requirements.txt / pyproject):

```
bot-commons @ git+https://github.com/jwojca/bot-commons.git@v0.1.0
```

## Použití

```python
from bot_commons import transcribe, record_usage, add_usage, format_usage
from bot_commons import parse_json_objects, parse_json_object

text = await transcribe(audio_bytes, api_key=OPENAI_API_KEY)            # OpenAI
text = await transcribe(audio_bytes, provider="local", local_url=URL)  # lokální whisper

usage = record_usage(response)                 # z Anthropic odpovědi
total = add_usage(usage_a, usage_b)            # akumulace napříč voláními
line = format_usage(total, show_cache=True)    # řádek do Telegramu

drafts = parse_json_objects(claude_raw)        # list[dict]
draft = parse_json_object(claude_raw)          # první dict
```

## Vývoj

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
