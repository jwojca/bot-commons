"""Parsování JSON z Claude odpovědí.

Umí: jeden objekt, JSON pole objektů, víc objektů za sebou (``{...}\\n{...}``),
markdown fence i preambuli/epilog kolem JSONu. Převzato z vyspělejší varianty
ve vylety-botu.
"""

from __future__ import annotations

import json


def parse_json_objects(raw: str) -> list[dict]:
    """Extrahuje všechny JSON objekty z textu a vrátí je jako seznam.

    Vždy vrací seznam dictů. Pokud se nenajde žádný validní objekt, vyhodí
    :class:`ValueError`.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    decoder = json.JSONDecoder()
    results: list[dict] = []
    i = 0
    n = len(text)
    while i < n:
        # přeskočit cokoli co není začátek objektu/pole
        while i < n and text[i] not in "{[":
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, list):
            results.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            results.append(obj)
        i = end

    if not results:
        raise ValueError(f"Claude nevrátil validní JSON: {raw[:300]}")
    return results


def parse_json_object(raw: str) -> dict:
    """Vrátí první JSON objekt z textu (convenience pro botery, kteří čekají jeden dict)."""
    return parse_json_objects(raw)[0]
