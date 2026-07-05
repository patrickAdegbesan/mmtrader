"""Performance metrics computed from a BacktestResult."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cognition.backtest.simulator import BacktestResult, Trade


def max_drawdown(equity_curve: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction (0.12 = -12%)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(-drawdown.min()) if len(drawdown) else 0.0


def sharpe_ratio(equity_curve: pd.Series, bars_per_year: int) -> float:
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])


def trade_stats(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "num_trades": 0, "win_rate": 0.0, "expectancy_r": 0.0,
            "avg_pnl": 0.0, "profit_factor": 0.0, "total_pnl": 0.0,
            "total_fees": 0.0, "total_slippage": 0.0, "avg_bars_held": 0.0,
            "avg_mae_r": 0.0,
        }
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gross_win = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    return {
        "num_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df),
        "expectancy_r": float(trades_df["r_multiple"].mean()),
        "avg_pnl": float(trades_df["pnl"].mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "total_pnl": float(trades_df["pnl"].sum()),
        "total_fees": float(trades_df["fees"].sum()),
        "total_slippage": float(trades_df["slippage_cost"].sum()),
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "avg_mae_r": float(trades_df["mae_r"].mean()),
    }


def compute_metrics(result: BacktestResult, bars_per_year: int) -> dict:
    trades_df = trades_to_frame(result.trades)
    stats = trade_stats(trades_df)
    stats.update({
        "initial_equity": result.initial_equity,
        "final_equity": result.final_equity,
        "total_return": result.final_equity / result.initial_equity - 1.0,
        "sharpe": sharpe_ratio(result.equity_curve, bars_per_year),
        "max_drawdown": max_drawdown(result.equity_curve),
    })
    return stats


def stats_by_group(trades_df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-regime or per-session trade statistics (spec: results must be
    reported separately for bull/bear/sideways).
    """
    if trades_df.empty or key not in trades_df.columns:
        return pd.DataFrame()
    rows = {}
    for group, sub in trades_df.groupby(key, dropna=True):
        rows[group] = trade_stats(sub)
    return pd.DataFrame(rows).T
