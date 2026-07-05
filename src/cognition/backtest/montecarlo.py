"""Monte Carlo robustness testing via randomized price-path perturbation.

Each run adds volatility-scaled gaussian noise to the bar-to-bar return
series, rebuilds a coherent OHLC path from it, and re-runs the strategy.
A robust strategy's edge should survive small perturbations; a curve-fit
one falls apart. Feature columns computed on the original path (e.g.
volatility_regime) are carried over unchanged — the perturbation is
deliberately small enough that regime labels remain representative.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from cognition.backtest.metrics import compute_metrics
from cognition.backtest.simulator import EventDrivenBacktester, Strategy


def perturb_ohlcv(
    df: pd.DataFrame,
    rng: np.random.Generator,
    noise_scale: float = 0.25,
    vol_window: int = 50,
) -> pd.DataFrame:
    """Return a copy of df with a perturbed price path. Noise per bar is
    gaussian with std = local rolling volatility * noise_scale, added to
    returns. OHLC bar shape is preserved by scaling each bar by the ratio
    of new close to old close.
    """
    out = df.copy()
    close = out["close"].to_numpy(dtype=float)
    returns = np.diff(close) / close[:-1]

    vol = pd.Series(returns).rolling(vol_window, min_periods=5).std()
    vol = vol.bfill().fillna(np.std(returns) if len(returns) else 0.0).to_numpy()

    noise = rng.normal(0.0, vol * noise_scale)
    new_returns = np.clip(returns + noise, -0.5, 0.5)
    new_close = np.empty_like(close)
    new_close[0] = close[0]
    new_close[1:] = close[0] * np.cumprod(1.0 + new_returns)

    scale = new_close / close
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].to_numpy(dtype=float) * scale
    return out


def run_monte_carlo(
    df: pd.DataFrame,
    engine: EventDrivenBacktester,
    strategy_factory: Callable[[], Strategy],
    runs: int,
    bars_per_year: int,
    noise_scale: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Run `runs` perturbed backtests; returns one metrics row per run."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(runs):
        perturbed = perturb_ohlcv(df, rng, noise_scale=noise_scale)
        result = engine.run(perturbed, strategy_factory())
        metrics = compute_metrics(result, bars_per_year)
        metrics["run"] = k
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_monte_carlo(mc_df: pd.DataFrame) -> dict:
    if mc_df.empty:
        return {}
    def pct(col: str, q: float) -> float:
        return float(mc_df[col].quantile(q))
    return {
        "runs": len(mc_df),
        "return_p5": pct("total_return", 0.05),
        "return_p50": pct("total_return", 0.50),
        "return_p95": pct("total_return", 0.95),
        "max_dd_p50": pct("max_drawdown", 0.50),
        "max_dd_p95": pct("max_drawdown", 0.95),
        "prob_profitable": float((mc_df["total_return"] > 0).mean()),
    }
