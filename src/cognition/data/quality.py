"""Data quality checks: never feed corrupted candles to the agents.

Three checks: missing candles (gaps), duplicate timestamps, and price
outliers (moves too large to be real, relative to recent volatility).
Callers get back a structured report plus a cleaned frame; deciding
whether to bail out on a bad report is the caller's call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cognition.config import DataQualityConfig


@dataclass
class GapReport:
    start_timestamp: int
    end_timestamp: int
    missing_candles: int


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    total_candles: int
    duplicate_timestamps: int
    gaps: list[GapReport] = field(default_factory=list)
    outlier_indices: list[int] = field(default_factory=list)

    @property
    def total_missing_candles(self) -> int:
        return sum(g.missing_candles for g in self.gaps)

    @property
    def is_clean(self) -> bool:
        return self.duplicate_timestamps == 0 and not self.gaps and not self.outlier_indices

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe}: {self.total_candles} candles, "
            f"{self.duplicate_timestamps} duplicates, {len(self.gaps)} gaps "
            f"({self.total_missing_candles} missing candles), "
            f"{len(self.outlier_indices)} outlier candidates"
        )


def find_duplicates(df: pd.DataFrame) -> int:
    return int(df["timestamp"].duplicated().sum())


def find_gaps(df: pd.DataFrame, timeframe_ms: int, max_gap_multiple: float) -> list[GapReport]:
    if len(df) < 2:
        return []
    diffs = df["timestamp"].diff().dropna()
    threshold = timeframe_ms * max_gap_multiple
    gap_mask = diffs > threshold
    gaps: list[GapReport] = []
    for idx in diffs[gap_mask].index:
        prev_ts = int(df.loc[idx - 1, "timestamp"])
        curr_ts = int(df.loc[idx, "timestamp"])
        missing = int(round((curr_ts - prev_ts) / timeframe_ms)) - 1
        gaps.append(GapReport(start_timestamp=prev_ts, end_timestamp=curr_ts, missing_candles=missing))
    return gaps


def find_outliers(df: pd.DataFrame, lookback: int, zscore_threshold: float) -> list[int]:
    if len(df) < lookback + 1:
        return []
    returns = df["close"].pct_change()
    rolling_std = returns.rolling(window=lookback, min_periods=lookback // 2).std()
    rolling_mean = returns.rolling(window=lookback, min_periods=lookback // 2).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscores = (returns - rolling_mean) / rolling_std
    outlier_mask = zscores.abs() > zscore_threshold
    return df.index[outlier_mask.fillna(False)].tolist()


def run_quality_checks(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    timeframe_ms: int,
    config: DataQualityConfig,
) -> QualityReport:
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    return QualityReport(
        symbol=symbol,
        timeframe=timeframe,
        total_candles=len(df_sorted),
        duplicate_timestamps=find_duplicates(df),
        gaps=find_gaps(df_sorted, timeframe_ms, config.max_gap_multiple),
        outlier_indices=find_outliers(df_sorted, config.outlier_lookback, config.outlier_zscore),
    )


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, dedupe, and drop rows with non-positive prices/volume — the
    minimum bar for "not corrupted". Gaps and outliers are reported, not
    silently dropped, since a real trader would want to see them.
    """
    cleaned = df.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    price_cols = ["open", "high", "low", "close"]
    valid = (cleaned[price_cols] > 0).all(axis=1) & (cleaned["volume"] >= 0)
    return cleaned[valid].reset_index(drop=True)
