"""Synthetic OHLCV generation for demos and tests.

Produces a price path that cycles through bull / sideways / bear /
sideways segments with occasional high-volatility patches, so regime
labeling, regime-split reporting, and the volatility-aware slippage
model all have something real to bite on — even in an environment where
the Bybit API is unreachable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cognition.utils.timeframes import timeframe_to_ms

SEGMENT_DRIFTS = {"bull": 0.00004, "sideways": 0.0, "bear": -0.00004}
SEGMENT_CYCLE = ["bull", "sideways", "bear", "sideways"]


def make_synthetic_ohlcv(
    n_bars: int = 30_000,
    timeframe: str = "1m",
    start_price: float = 50_000.0,
    start_timestamp: int = 1_650_000_000_000,
    base_vol: float = 0.0008,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tf_ms = timeframe_to_ms(timeframe)
    seg_len = max(n_bars // 8, 500)

    drift = np.zeros(n_bars)
    for k in range(0, n_bars, seg_len):
        segment = SEGMENT_CYCLE[(k // seg_len) % len(SEGMENT_CYCLE)]
        drift[k : k + seg_len] = SEGMENT_DRIFTS[segment]

    # Random high-volatility patches (~10% of bars at 3x volatility).
    vol = np.full(n_bars, base_vol)
    n_patches = max(n_bars // 5000, 1)
    for _ in range(n_patches):
        p_start = rng.integers(0, max(n_bars - 500, 1))
        vol[p_start : p_start + 500] = base_vol * 3

    returns = drift + rng.normal(0, vol)
    close = start_price * np.cumprod(1 + returns)
    open_ = np.empty_like(close)
    open_[0] = start_price
    open_[1:] = close[:-1]

    wick = np.abs(rng.normal(0, vol)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.uniform(1, 20, n_bars) * (1 + 5 * (vol / base_vol - 1))

    return pd.DataFrame(
        {
            "timestamp": start_timestamp + np.arange(n_bars, dtype=np.int64) * tf_ms,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
