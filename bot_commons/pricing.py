"""Výpočet a formátování ceny za tokeny z Anthropic odpovědí.

Sjednocuje dvě dřívější varianty:
- jidlo-bot: per-model breakdown + cena v CZK,
- vylety-bot: cache-aware tokeny (cache_creation/cache_read).

Vnitřní tvar usage::

    {"breakdown": [
        {"model", "input", "output", "cache_creation", "cache_read", "cost_czk"},
        ...,
    ]}

Funkce čtou tokeny z odpovědi přes ``getattr`` – žádná závislost na ``anthropic`` SDK.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

USD_TO_CZK = 22.5

# USD za milion tokenů: (input, output)
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}
_DEFAULT_PRICE = (3.00, 15.00)

# Multiplikátory ceny vstupu pro cache (Anthropic ephemeral cache).
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.1

# Krátké názvy pro zobrazení.
_SHORT = {
    "claude-sonnet-4-6": "Sonnet",
    "claude-haiku-4-5-20251001": "Haiku",
}

_KEYS = ("input", "output", "cache_creation", "cache_read")


def _short(model: str) -> str:
    return _SHORT.get(model, model)


def _cost_czk(entry: dict[str, Any], model: str, usd_to_czk: float) -> float:
    price_in, price_out = PRICING.get(model, _DEFAULT_PRICE)
    tokens_usd = (
        entry["input"] * price_in
        + entry["output"] * price_out
        + entry["cache_creation"] * price_in * _CACHE_WRITE_MULT
        + entry["cache_read"] * price_in * _CACHE_READ_MULT
    )
    return tokens_usd / 1_000_000 * usd_to_czk


def record_usage(
    response: Any, model: str | None = None, *, usd_to_czk: float = USD_TO_CZK
) -> dict[str, Any]:
    """Vytáhne tokeny z Anthropic odpovědi a vrátí usage s jedním breakdown řádkem."""
    u = getattr(response, "usage", None)
    model = model or getattr(response, "model", "?")
    entry = {
        "model": model,
        "input": getattr(u, "input_tokens", 0) or 0,
        "output": getattr(u, "output_tokens", 0) or 0,
        "cache_creation": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
    }
    entry["cost_czk"] = _cost_czk(entry, model, usd_to_czk)
    log.info(
        "Tokens: in=%d out=%d cache_write=%d cache_read=%d model=%s cost=%.3f Kč",
        entry["input"], entry["output"], entry["cache_creation"],
        entry["cache_read"], _short(model), entry["cost_czk"],
    )
    return {"breakdown": [entry]}


def add_usage(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any]:
    """Sloučí dvě usage dicts; zachová per-model breakdown, sečte tokeny i cenu."""
    entries: dict[str, dict[str, Any]] = {}
    for u in (a, b):
        if not u:
            continue
        for e in u.get("breakdown", []):
            m = e["model"]
            if m in entries:
                for k in _KEYS:
                    entries[m][k] += e.get(k, 0)
                entries[m]["cost_czk"] += e.get("cost_czk", 0.0)
            else:
                entries[m] = {"model": m, "cost_czk": e.get("cost_czk", 0.0)}
                for k in _KEYS:
                    entries[m][k] = e.get(k, 0)
    return {"breakdown": list(entries.values())}


def format_usage(usage: dict[str, Any] | None, *, show_cache: bool = False) -> str:
    """Vrátí lidsky čitelný řádek pro zobrazení v Telegramu."""
    if not usage:
        return ""
    breakdown = usage.get("breakdown", [])
    if not breakdown:
        return ""

    total_cost = sum(e.get("cost_czk", 0.0) for e in breakdown)
    if len(breakdown) == 1:
        e = breakdown[0]
        base = f"📊 Tokeny: {e['input']} in / {e['output']} out · {_short(e['model'])}"
        if show_cache and (e.get("cache_creation") or e.get("cache_read")):
            base += f" (cache {e['cache_read']} read / {e['cache_creation']} write)"
        parts = [base]
    else:
        token_parts = " + ".join(
            f"{_short(e['model'])} {e['input']}/{e['output']}" for e in breakdown
        )
        parts = [f"📊 Tokeny: {token_parts}"]
    if total_cost:
        parts.append(f"~{total_cost:.2f} Kč")
    return " · ".join(parts)
