"""Strategies for exercising the backtester.

SmaCrossoverStrategy is the Milestone-2 "dummy strategy" — it exists to
prove the engine, cost model, and reporting work end-to-end, NOT to make
money. The RL agents replace it from Milestone 3 onward.

ScriptedStrategy emits pre-planned signals at exact bar indices; used by
the test suite to verify engine mechanics deterministically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cognition.backtest.simulator import Signal


class ScriptedStrategy:
    def __init__(self, signals: dict[int, Signal]):
        self._signals = dict(signals)

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def prepare(self, df: pd.DataFrame) -> None:
        pass

    def signal(self, i: int) -> Signal | None:
        return self._signals.get(i)


class SmaCrossoverStrategy:
    """Long on fast-SMA crossing above slow-SMA, short on crossing below.
    Fixed fractional stop/target. Opposite crossover also closes an open
    position (the engine handles that via the direction mismatch).
    """

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        stop_loss_pct: float = 0.004,
        take_profit_pct: float = 0.008,
        max_holding_bars: int | None = 240,
    ):
        self.fast = fast
        self.slow = slow
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_holding_bars = max_holding_bars
        self._cross: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        # The dummy strategy has nothing to learn; RL agents will use this
        # hook for real in Milestone 3.
        pass

    def prepare(self, df: pd.DataFrame) -> None:
        close = df["close"]
        fast_ma = close.rolling(self.fast).mean()
        slow_ma = close.rolling(self.slow).mean()
        sign = np.sign((fast_ma - slow_ma).to_numpy())
        cross = np.zeros(len(df), dtype=int)
        valid = ~np.isnan(sign)
        for i in range(1, len(df)):
            if valid[i] and valid[i - 1] and sign[i] != sign[i - 1] and sign[i] != 0:
                cross[i] = int(sign[i])
        self._cross = cross

    def signal(self, i: int) -> Signal | None:
        direction = int(self._cross[i])
        if direction == 0:
            return None
        return Signal(
            direction=direction,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            max_holding_bars=self.max_holding_bars,
        )
