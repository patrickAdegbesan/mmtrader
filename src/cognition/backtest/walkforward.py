"""Walk-forward validation harness and the out-of-sample lockbox split.

Walk-forward: train on window 1 → test on window 2 → roll forward. Each
test window is prefixed with `warmup_bars` taken from the end of its
train window so indicators have history, but the engine is told (via
start_index) not to trade inside the warmup.

Lockbox: the most recent N months are split off before ANY training or
tuning ever sees them. Only final validation may touch the lockbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import pandas as pd

from cognition.backtest.metrics import compute_metrics, trades_to_frame
from cognition.backtest.simulator import BacktestResult, EventDrivenBacktester, Strategy
from cognition.utils.timeframes import MS_PER_DAY


def split_lockbox(df: pd.DataFrame, months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off the most recent `months` (30-day months) as the lockbox."""
    cutoff = int(df["timestamp"].max()) - months * 30 * MS_PER_DAY
    main = df[df["timestamp"] < cutoff].reset_index(drop=True)
    lockbox = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return main, lockbox


def walk_forward_splits(
    df: pd.DataFrame, train_bars: int, test_bars: int, warmup_bars: int = 0
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Yield (train_df, test_df_with_warmup, start_index) tuples rolling
    forward by test_bars each step. Test windows never overlap.
    """
    warmup_bars = min(warmup_bars, train_bars)
    n = len(df)
    start = 0
    while start + train_bars + test_bars <= n:
        train = df.iloc[start : start + train_bars].reset_index(drop=True)
        test_begin = start + train_bars - warmup_bars
        test = df.iloc[test_begin : start + train_bars + test_bars].reset_index(drop=True)
        yield train, test, warmup_bars
        start += test_bars


@dataclass
class WindowResult:
    window: int
    train_start: int
    test_start: int
    test_end: int
    metrics: dict
    result: BacktestResult


def run_walk_forward(
    df: pd.DataFrame,
    engine: EventDrivenBacktester,
    strategy_factory: Callable[[], Strategy],
    train_bars: int,
    test_bars: int,
    warmup_bars: int,
    bars_per_year: int,
) -> tuple[list[WindowResult], pd.DataFrame]:
    """Run the full walk-forward. Returns per-window results and the
    combined out-of-sample trade frame (all test-window trades pooled).
    """
    windows: list[WindowResult] = []
    all_trades = []
    for k, (train, test, start_index) in enumerate(
        walk_forward_splits(df, train_bars, test_bars, warmup_bars)
    ):
        strategy = strategy_factory()
        strategy.fit(train)
        result = engine.run(test, strategy, start_index=start_index)
        windows.append(
            WindowResult(
                window=k,
                train_start=int(train["timestamp"].iloc[0]),
                test_start=int(test["timestamp"].iloc[start_index]),
                test_end=int(test["timestamp"].iloc[-1]),
                metrics=compute_metrics(result, bars_per_year),
                result=result,
            )
        )
        all_trades.extend(result.trades)
    return windows, trades_to_frame(all_trades)
