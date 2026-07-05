"""Market regime labeling: bull / bear / sideways.

Uses the *trailing* return over a rolling window — no lookahead, so the
label at bar i is known at bar i and can safely be fed to agents later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BULL, BEAR, SIDEWAYS, UNKNOWN = "bull", "bear", "sideways", "unknown"


def label_market_regimes(df: pd.DataFrame, window_bars: int, threshold: float = 0.03) -> pd.DataFrame:
    """Add a `market_regime` column. Trailing return over `window_bars`
    above +threshold → bull, below −threshold → bear, else sideways.
    The first `window_bars` rows are labeled "unknown".
    """
    out = df.copy()
    trailing_return = out["close"] / out["close"].shift(window_bars) - 1.0
    out["market_regime"] = np.select(
        [trailing_return > threshold, trailing_return < -threshold],
        [BULL, BEAR],
        default=SIDEWAYS,
    )
    out.loc[trailing_return.isna(), "market_regime"] = UNKNOWN
    return out
