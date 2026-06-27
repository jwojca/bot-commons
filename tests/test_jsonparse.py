"""Testy parsování JSON z Claude odpovědí."""

from __future__ import annotations

import pytest

from bot_commons import jsonparse


def test_single_object():
    assert jsonparse.parse_json_objects('{"a": 1}') == [{"a": 1}]


def test_markdown_fence():
    raw = '```json\n{"a": 1, "b": "x"}\n```'
    assert jsonparse.parse_json_objects(raw) == [{"a": 1, "b": "x"}]


def test_array_of_objects():
    assert jsonparse.parse_json_objects('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]


def test_multiple_objects_back_to_back():
    raw = '{"a": 1}\n{"a": 2}'
    assert jsonparse.parse_json_objects(raw) == [{"a": 1}, {"a": 2}]


def test_preamble_and_epilog():
    raw = 'Tady je výsledek:\n{"a": 1}\nHotovo.'
    assert jsonparse.parse_json_objects(raw) == [{"a": 1}]


def test_parse_json_object_returns_first():
    assert jsonparse.parse_json_object('{"a": 1}\n{"a": 2}') == {"a": 1}


def test_no_json_raises():
    with pytest.raises(ValueError):
        jsonparse.parse_json_objects("žádný json tu není")
