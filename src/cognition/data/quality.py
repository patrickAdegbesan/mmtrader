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
    zero_volume_candles: int = 0
    stale_candles: int = 0
    max_outlier_fraction: float = 0.005
    max_zero_volume_fraction: float = 0.01

    @property
    def total_missing_candles(self) -> int:
        return sum(g.missing_candles for g in self.gaps)

    @property
    def outlier_fraction(self) -> float:
        return len(self.outlier_indices) / self.total_candles if self.total_candles else 0.0

    @property
    def zero_volume_fraction(self) -> float:
        return self.zero_volume_candles / self.total_candles if self.total_candles else 0.0

    @property
    def has_real_activity(self) -> bool:
        """False when too many candles show no trading at all — the
        signature of testnet data or a dead symbol, both unusable for
        training even though every candle is individually well-formed.
        """
        return self.zero_volume_fraction <= self.max_zero_volume_fraction

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicate_timestamps == 0
            and not self.gaps
            and self.outlier_fraction <= self.max_outlier_fraction
            and self.has_real_activity
        )

    def summary(self) -> str:
        return (
            f"{self.symbol} {self.timeframe}: {self.total_candles} candles, "
            f"{self.duplicate_timestamps} duplicates, {len(self.gaps)} gaps "
            f"({self.total_missing_candles} missing candles), "
            f"{len(self.outlier_indices)} outlier candidates "
            f"({self.outlier_fraction:.2%}), {self.zero_volume_candles} zero-volume "
            f"({self.zero_volume_fraction:.1%}), {self.stale_candles} stale"
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


def find_zero_volume(df: pd.DataFrame) -> int:
    """Candles where nothing traded. A handful is normal on an illiquid
    pair; a large fraction means the feed isn't a real market.
    """
    if "volume" not in df.columns:
        return 0
    return int((df["volume"] <= 0).sum())


def find_stale(df: pd.DataFrame) -> int:
    """Candles with no intra-bar movement at all (open == high == low ==
    close) — a synthetic-feed signature that gap and outlier checks miss.
    """
    price_cols = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in price_cols):
        return 0
    return int((df[price_cols].nunique(axis=1) == 1).sum())


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
        zero_volume_candles=find_zero_volume(df_sorted),
        stale_candles=find_stale(df_sorted),
        max_outlier_fraction=config.max_outlier_fraction,
        max_zero_volume_fraction=config.max_zero_volume_fraction,
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
