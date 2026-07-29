"""Feature extraction: turns raw OHLCV candles into the signals the
agents will train on.

Note on order-book imbalance / spread width: these require live
order-book snapshots, which historical OHLCV candles do not contain.
The columns are still emitted (as NaN) here so downstream schemas stay
stable; they get populated once the live WebSocket order-book feed
lands (a later milestone) and are backfilled where historical
order-book data is available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import BollingerBands

# UTC-hour session buckets. Real trading sessions overlap; this is a
# simplified non-overlapping tagging scheme so every candle gets exactly
# one session label for regime-memory purposes (see architecture doc §7).
SESSION_BOUNDARIES = [
    (0, 8, "asian"),
    (8, 16, "european"),
    (16, 24, "us"),
]

VOLATILITY_WINDOW = 50
VOLATILITY_QUANTILES = (0.33, 0.66)


def add_price_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    df["price_velocity"] = df["close"].pct_change()
    df["price_acceleration"] = df["price_velocity"].diff()
    return df


def add_trailing_momentum(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """N-bar trailing return. 1-bar price_velocity is dominated by bid/ask
    bounce noise at this timeframe (Spearman rho vs forward returns ~0.01
    on real BTC 1m data); a ~15-bar trailing return is where the real,
    cost-clearing-magnitude signal shows up (rho ~0.045, same sign and
    order of magnitude as RSI/Bollinger/Donchian's mean-reversion read) —
    see the diagnostic run backing this change.
    """
    df[f"trailing_return_{window}"] = df["close"] / df["close"].shift(window) - 1.0
    return df


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_9"] = SMAIndicator(close=df["close"], window=9).sma_indicator()
    df["sma_21"] = SMAIndicator(close=df["close"], window=21).sma_indicator()
    df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    return df


def add_bollinger_position(df: pd.DataFrame) -> pd.DataFrame:
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    upper, lower = bb.bollinger_hband(), bb.bollinger_lband()
    band_width = upper - lower
    df["bollinger_position"] = np.where(band_width > 0, (df["close"] - lower) / band_width, np.nan)
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volume_delta"] = df["volume"].diff()
    rolling = df["volume"].rolling(window=50, min_periods=25)
    std = rolling.std()
    df["volume_zscore"] = np.where(std > 0, (df["volume"] - rolling.mean()) / std, 0.0)
    return df


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    df["sma21_distance"] = df["close"] / df["sma_21"] - 1.0
    return df


def add_breakout_features(df: pd.DataFrame) -> pd.DataFrame:
    # Position of close within the trailing 20-bar Donchian channel; >1 or <0
    # never happens (close is part of the window), 1.0 = at the highs.
    hi = df["high"].rolling(20).max()
    lo = df["low"].rolling(20).min()
    channel = hi - lo
    df["donchian_position"] = np.where(channel > 0, (df["close"] - lo) / channel, 0.5)
    return df


def add_microstructure_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Candle-derived stand-ins for order-book features that historical
    OHLCV cannot provide. `range_pct` proxies spread/liquidity thinness;
    `close_in_range` proxies buy/sell imbalance. The Microstructure agent
    trains on these until the live order-book WebSocket feed (Milestone 6)
    populates `order_book_imbalance` and `spread_width` for real.
    """
    bar_range = df["high"] - df["low"]
    df["range_pct"] = bar_range / df["close"]
    df["close_in_range"] = np.where(bar_range > 0, (df["close"] - df["low"]) / bar_range, 0.5)
    return df


def add_microstructure_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    df["order_book_imbalance"] = np.nan
    df["spread_width"] = np.nan
    return df


def add_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    returns = df["close"].pct_change()
    rolling_vol = returns.rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW // 2).std()
    df["volatility"] = rolling_vol

    low_q, high_q = rolling_vol.quantile(VOLATILITY_QUANTILES[0]), rolling_vol.quantile(VOLATILITY_QUANTILES[1])
    conditions = [rolling_vol <= low_q, rolling_vol >= high_q]
    choices = ["low", "high"]
    df["volatility_regime"] = np.select(conditions, choices, default="medium")
    df.loc[rolling_vol.isna(), "volatility_regime"] = np.nan
    return df


def _session_for_hour(hour: int) -> str:
    for start, end, label in SESSION_BOUNDARIES:
        if start <= hour < end:
            return label
    return "unknown"


def add_session_tag(df: pd.DataFrame) -> pd.DataFrame:
    hours = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.hour
    df["session"] = hours.map(_session_for_hour)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline. Expects a sorted OHLCV frame with
    columns [timestamp, open, high, low, close, volume].
    """
    out = df.copy()
    out = add_price_dynamics(out)
    out = add_trailing_momentum(out)
    out = add_momentum_indicators(out)
    out = add_moving_averages(out)
    out = add_bollinger_position(out)
    out = add_volume_features(out)
    out = add_mean_reversion_features(out)
    out = add_breakout_features(out)
    out = add_microstructure_proxies(out)
    out = add_microstructure_placeholders(out)
    out = add_volatility_regime(out)
    out = add_session_tag(out)
    return out
