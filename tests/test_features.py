from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cognition.features.engineering import extract_features, _session_for_hour

TIMEFRAME_MS = 60_000


def make_df(n=300, start_ts=1_700_000_000_000):
    timestamps = [start_ts + i * TIMEFRAME_MS for i in range(n)]
    prices = 50_000 + np.cumsum(np.random.default_rng(7).normal(0, 1, n))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": np.random.default_rng(2).uniform(1, 10, n),
        }
    )


def test_extract_features_adds_expected_columns():
    df = make_df()
    out = extract_features(df)
    expected = {
        "price_velocity", "price_acceleration", "rsi_14", "macd", "macd_signal",
        "macd_diff", "sma_9", "sma_21", "ema_9", "ema_21", "bollinger_position",
        "volume_delta", "order_book_imbalance", "spread_width", "volatility",
        "volatility_regime", "session",
    }
    assert expected.issubset(out.columns)
    assert len(out) == len(df)


def test_extract_features_does_not_mutate_input():
    df = make_df()
    original = df.copy()
    extract_features(df)
    pd.testing.assert_frame_equal(df, original)


def test_bollinger_position_within_reasonable_bounds():
    df = make_df()
    out = extract_features(df)
    valid = out["bollinger_position"].dropna()
    # Not a hard mathematical guarantee (price can pierce the bands), but
    # for smooth synthetic data it should stay close to [0, 1].
    assert valid.between(-1, 2).all()


def test_session_tagging_matches_utc_hour_buckets():
    assert _session_for_hour(0) == "asian"
    assert _session_for_hour(7) == "asian"
    assert _session_for_hour(8) == "european"
    assert _session_for_hour(15) == "european"
    assert _session_for_hour(16) == "us"
    assert _session_for_hour(23) == "us"


def test_session_column_reflects_actual_candle_hour():
    ts = int(datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)
    df = make_df(n=5, start_ts=ts)
    out = extract_features(df)
    assert (out["session"] == "european").all()


def test_microstructure_placeholders_are_nan():
    df = make_df()
    out = extract_features(df)
    assert out["order_book_imbalance"].isna().all()
    assert out["spread_width"].isna().all()
