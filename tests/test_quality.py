import numpy as np
import pandas as pd
import pytest

from cognition.config import DataQualityConfig
from cognition.data.quality import (
    clean_ohlcv,
    find_duplicates,
    find_gaps,
    find_outliers,
    run_quality_checks,
)

TIMEFRAME_MS = 60_000  # 1m


def make_clean_df(n=300, start_ts=1_700_000_000_000, price=50_000.0):
    timestamps = [start_ts + i * TIMEFRAME_MS for i in range(n)]
    prices = price + np.cumsum(np.random.default_rng(42).normal(0, 1, n))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.random.default_rng(1).uniform(1, 10, n),
        }
    )


def test_find_duplicates_detects_repeated_timestamps():
    df = make_clean_df(10)
    dup_row = df.iloc[[3]]
    df_with_dupe = pd.concat([df, dup_row], ignore_index=True)
    assert find_duplicates(df_with_dupe) == 1
    assert find_duplicates(df) == 0


def test_find_gaps_detects_missing_candles():
    df = make_clean_df(20)
    # remove 5 consecutive candles in the middle to create a gap
    df_with_gap = pd.concat([df.iloc[:10], df.iloc[15:]], ignore_index=True)
    gaps = find_gaps(df_with_gap, TIMEFRAME_MS, max_gap_multiple=2)
    assert len(gaps) == 1
    assert gaps[0].missing_candles == 5


def test_find_gaps_no_false_positive_on_contiguous_data():
    df = make_clean_df(50)
    assert find_gaps(df, TIMEFRAME_MS, max_gap_multiple=2) == []


def test_find_outliers_flags_extreme_move():
    df = make_clean_df(300)
    spike_idx = 250
    df.loc[spike_idx, "close"] = df.loc[spike_idx - 1, "close"] * 5  # +400% in one candle
    outliers = find_outliers(df, lookback=200, zscore_threshold=6.0)
    assert spike_idx in outliers


def test_find_outliers_empty_on_short_series():
    df = make_clean_df(10)
    assert find_outliers(df, lookback=200, zscore_threshold=6.0) == []


def test_run_quality_checks_reports_clean_data():
    df = make_clean_df(300)
    report = run_quality_checks(df, "BTC/USDT", "1m", TIMEFRAME_MS, DataQualityConfig())
    assert report.is_clean
    assert report.total_candles == 300


def test_clean_ohlcv_drops_duplicates_and_invalid_rows():
    df = make_clean_df(10)
    dup_row = df.iloc[[2]]
    df_dirty = pd.concat([df, dup_row], ignore_index=True)
    df_dirty.loc[5, "close"] = -1  # invalid price
    cleaned = clean_ohlcv(df_dirty)
    assert find_duplicates(cleaned) == 0
    assert (cleaned["close"] > 0).all()
    assert len(cleaned) == len(df) - 1  # the invalid row was dropped
