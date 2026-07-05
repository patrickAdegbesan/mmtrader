"""Timeframe string helpers ("1m", "5m", "1h", ...) without needing a
live ccxt exchange object.
"""
from __future__ import annotations

_UNIT_MS = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}

MS_PER_YEAR = 365 * 86_400_000
MS_PER_DAY = 86_400_000


def timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    if unit not in _UNIT_MS:
        raise ValueError(f"Unsupported timeframe unit: {timeframe!r}")
    return int(timeframe[:-1]) * _UNIT_MS[unit]


def bars_per_day(timeframe: str) -> int:
    return MS_PER_DAY // timeframe_to_ms(timeframe)


def bars_per_year(timeframe: str) -> int:
    return MS_PER_YEAR // timeframe_to_ms(timeframe)
