"""Testy výpočtu a formátování ceny tokenů."""

from __future__ import annotations

from types import SimpleNamespace

from bot_commons import pricing


def _resp(model, input_tokens, output_tokens, cache_creation=0, cache_read=0):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )
    return SimpleNamespace(model=model, usage=usage)


def test_record_usage_basic_cost():
    u = pricing.record_usage(_resp("claude-sonnet-4-6", 1_000_000, 1_000_000))
    entry = u["breakdown"][0]
    # 1M in * 3 + 1M out * 15 = 18 USD; * 22.5 = 405 Kč
    assert entry["cost_czk"] == 405.0
    assert entry["model"] == "claude-sonnet-4-6"
    assert entry["input"] == 1_000_000
    assert entry["output"] == 1_000_000


def test_record_usage_model_override_and_fallback_to_response():
    # model z odpovědi
    u = pricing.record_usage(_resp("claude-haiku-4-5-20251001", 0, 0))
    assert u["breakdown"][0]["model"] == "claude-haiku-4-5-20251001"
    # explicitní override
    u2 = pricing.record_usage(_resp("ignored", 0, 0), model="claude-sonnet-4-6")
    assert u2["breakdown"][0]["model"] == "claude-sonnet-4-6"


def test_record_usage_cache_pricing():
    # 1M cache_read na sonnet = 1M * 3 * 0.1 / 1M = 0.3 USD * 22.5 = 6.75 Kč
    u = pricing.record_usage(_resp("claude-sonnet-4-6", 0, 0, cache_read=1_000_000))
    assert round(u["breakdown"][0]["cost_czk"], 2) == 6.75
    # 1M cache_write = 1M * 3 * 1.25 / 1M = 3.75 USD * 22.5 = 84.375
    u2 = pricing.record_usage(_resp("claude-sonnet-4-6", 0, 0, cache_creation=1_000_000))
    assert round(u2["breakdown"][0]["cost_czk"], 3) == 84.375


def test_record_usage_handles_missing_usage():
    u = pricing.record_usage(SimpleNamespace(model="claude-sonnet-4-6", usage=None))
    e = u["breakdown"][0]
    assert e["input"] == 0 and e["output"] == 0 and e["cost_czk"] == 0.0


def test_add_usage_merges_same_model():
    a = pricing.record_usage(_resp("claude-sonnet-4-6", 100, 200))
    b = pricing.record_usage(_resp("claude-sonnet-4-6", 50, 50))
    merged = pricing.add_usage(a, b)
    assert len(merged["breakdown"]) == 1
    e = merged["breakdown"][0]
    assert e["input"] == 150
    assert e["output"] == 250
    assert round(e["cost_czk"], 6) == round(
        a["breakdown"][0]["cost_czk"] + b["breakdown"][0]["cost_czk"], 6
    )


def test_add_usage_keeps_per_model_breakdown():
    a = pricing.record_usage(_resp("claude-sonnet-4-6", 100, 100))
    b = pricing.record_usage(_resp("claude-haiku-4-5-20251001", 100, 100))
    merged = pricing.add_usage(a, b)
    models = {e["model"] for e in merged["breakdown"]}
    assert models == {"claude-sonnet-4-6", "claude-haiku-4-5-20251001"}


def test_add_usage_with_none():
    a = pricing.record_usage(_resp("claude-sonnet-4-6", 10, 10))
    assert pricing.add_usage(a, None)["breakdown"][0]["input"] == 10
    assert pricing.add_usage(None, None) == {"breakdown": []}


def test_format_usage_empty():
    assert pricing.format_usage(None) == ""
    assert pricing.format_usage({"breakdown": []}) == ""


def test_format_usage_single_model():
    u = pricing.record_usage(_resp("claude-sonnet-4-6", 1200, 340))
    s = pricing.format_usage(u)
    assert "1200 in / 340 out" in s
    assert "Sonnet" in s
    assert "Kč" in s


def test_format_usage_show_cache():
    u = pricing.record_usage(_resp("claude-sonnet-4-6", 100, 50, cache_read=900, cache_creation=200))
    s = pricing.format_usage(u, show_cache=True)
    assert "cache 900 read / 200 write" in s


def test_format_usage_multi_model():
    merged = pricing.add_usage(
        pricing.record_usage(_resp("claude-sonnet-4-6", 100, 100)),
        pricing.record_usage(_resp("claude-haiku-4-5-20251001", 200, 200)),
    )
    s = pricing.format_usage(merged)
    assert "Sonnet 100/100" in s
    assert "Haiku 200/200" in s
