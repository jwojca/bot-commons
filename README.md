# bot-commons

Sdílená logika napříč mými Telegram boty (`crm-bot`, `jidlo-bot`, `vylety-bot`).
Jedno místo pravdy pro věci, které se dřív kopírovaly a rozjížděly.

## Co obsahuje

| Modul | Co dělá |
|---|---|
| `bot_commons.transcription` | Přepis hlasu – OpenAI Whisper, Gemini (multimodální audio) nebo lokální whisper služba, přepínatelné konfigurací. |
| `bot_commons.whisper` | Zpětně kompatibilní `transcribe()` fasáda nad `transcription` (vrací plain `str`). |
| `bot_commons.pricing` | Výpočet ceny za tokeny z Anthropic odpovědi (per-model breakdown, cache, CZK) + formátování. |
| `bot_commons.jsonparse` | Parsování JSON z Claude odpovědí (markdown fence, pole, víc objektů za sebou). |
| `bot_commons.config` | Lehké env/logging helpery (`require_env`, `env_flag`, `setup_logging`). |

Funkce berou vstupy **explicitně** (api_key, model, …) – nečtou žádné globály,
takže si je každý bot obalí ve svém stylu. Jediná výjimka je opt-in
`provider_from_env()`, kterou si zavoláš, jen když chceš přepis přepínat
přes prostředí.

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
text = await transcribe(audio_bytes, api_key=GEMINI_API_KEY, provider="gemini")
text = await transcribe(audio_bytes, provider="local", local_url=URL)  # lokální whisper

usage = record_usage(response)                 # z Anthropic odpovědi
total = add_usage(usage_a, usage_b)            # akumulace napříč voláními
line = format_usage(total, show_cache=True)    # řádek do Telegramu

drafts = parse_json_objects(claude_raw)        # list[dict]
draft = parse_json_object(claude_raw)          # první dict
```

## Přepis hlasu: výběr providera

Tři providery za jedním rozhraním (`TranscriptionProvider`). Přepíná se
konfigurací, ne zásahem do kódu.

| `TRANSCRIPTION_PROVIDER` | Co běží | Povinné env proměnné | Volitelné |
|---|---|---|---|
| `openai` (výchozí) | OpenAI Whisper API | `OPENAI_API_KEY` | `WHISPER_MODEL` (default `whisper-1`) |
| `gemini` | Gemini API, multimodální audio vstup | `GEMINI_API_KEY` | `GEMINI_MODEL` (default `gemini-3.6-flash`) |
| `local` | Vlastní whisper služba | `WHISPER_LOCAL_URL` | – |

Sestaví se **jen zvolený** provider, takže klíč toho druhého nastavený být nemusí.

```bash
# .env – OpenAI varianta
TRANSCRIPTION_PROVIDER=openai
OPENAI_API_KEY=sk-...

# .env – Gemini varianta
TRANSCRIPTION_PROVIDER=gemini
GEMINI_API_KEY=AIza...
GEMINI_MODEL=gemini-3.6-flash   # volitelné
```

```python
from bot_commons import provider_from_env, build_provider

provider = provider_from_env()                     # podle TRANSCRIPTION_PROVIDER
result = await provider.transcribe(audio_bytes)
result.text, result.provider, result.model         # stejný tvar u všech providerů

# Explicitně / per request, bez ohledu na env:
provider = build_provider("gemini", api_key=KEY, model="gemini-2.5-pro")
```

`provider_from_env()` je **jediné** místo v knihovně, které čte env proměnné –
zbytek bere všechno explicitně. Kdo chce konfiguraci řešit po svém, použije
`build_provider()` a env vůbec nepotřebuje.

Poznámky k chování:

- **Žádný tichý fallback.** Když zvolený provider selže (timeout, chyba API,
  chybějící klíč), letí `TranscriptionError` s názvem providera v hlášce a chyba
  se zaloguje. Na druhého providera se nikdy nepřepne samo.
- Chybějící klíč spadne **při sestavení** providera, ne až u prvního přepisu.
- Gemini posílá audio inline jako base64; nad 12 MiB se přepne na Files API.
  Telegram hlasovky se do inline vejdou vždycky.
- **Google modely stahuje z oběhu.** Když API vrátí 404 s „no longer available",
  je čas přepnout `GEMINI_MODEL`. Aktuální seznam:
  `curl -H "x-goog-api-key: $GEMINI_API_KEY" \
  https://generativelanguage.googleapis.com/v1beta/models`
- Gemini dostává prompt na doslovný přepis s jazykovým hintem (`language="cs"`
  → „Transcribe this audio verbatim in Czech“).

## Vývoj

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
