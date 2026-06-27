"""Lehké helpery na načítání env proměnných a logging.

Cílí hlavně na boty s modulovými globály (jidlo/vylety). Boti s vlastní
bohatší konfigurací (crm-bot) si nechávají to své a sáhnou si max. po
``env_flag``/``require_env``.
"""

from __future__ import annotations

import logging
import os

_DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class ConfigError(RuntimeError):
    """Povinná env proměnná chybí."""


def require_env(name: str) -> str:
    """Vrátí hodnotu env proměnné, nebo vyhodí :class:`ConfigError` s jasnou hláškou."""
    value = os.environ.get(name)
    if value is None or value == "":
        raise ConfigError(f"Chybí povinná env proměnná: {name}")
    return value


def get_env(name: str, default: str | None = None) -> str | None:
    """Vrátí hodnotu env proměnné, nebo ``default``."""
    return os.environ.get(name, default)


def env_flag(name: str, default: bool = False) -> bool:
    """Booleovská env proměnná: ``"1"/"true"/"yes"/"on"`` (case-insensitive) → True."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def setup_logging(level: str = "INFO", *, fmt: str = _DEFAULT_LOG_FORMAT) -> None:
    """Nastaví ``logging.basicConfig`` ve sdíleném formátu."""
    logging.basicConfig(level=level, format=fmt)
