"""Gym-style scalping environment stepping bar-by-bar over feature-
enriched OHLCV, applying the SAME CostModel arithmetic as the Milestone-2
backtester (fees, adverse regime-aware slippage, next-bar execution).

Honesty rules mirrored from the backtester:
- An action chosen at bar i's close executes at bar i+1's open.
- Intra-bar: stop assumed hit before target; gaps fill at the worse open.
- Sizing risks `risk_per_trade` of episode equity off the stop distance.

Reward is risk-adjusted (the spec's non-negotiable): the R-multiple of
the closed trade (PnL / capital risked, after all costs), plus a small
per-bar holding penalty to discourage aimless exposure. No reward for
raw unrealized profit — agents get paid on completed, cost-adjusted
outcomes only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cognition.agents.actions import FLAT, ActionMapper
from cognition.backtest.costs import CostModel

# Feature columns fed to the momentum agent's state vector.
MOMENTUM_FEATURES = [
    "price_velocity",
    "price_acceleration",
    "rsi_14",
    "macd_diff",
    "bollinger_position",
    "volume_delta",
    "volatility",
]
HOLDING_PENALTY = 0.002  # per-bar, in R units


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "FeatureStats":
        values = df[columns].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(mean=np.array(d["mean"]), std=np.array(d["std"]))


@dataclass
class _EnvPosition:
    direction: int
    entry_price: float
    size: float
    stop_price: float
    tp_price: float
    stop_distance: float
    entry_fee: float
    entry_index: int


class ScalpingEnv:
    """Observation = normalized feature vector + [position_dir, unrealized_R, bars_held/100]."""

    def __init__(
        self,
        df: pd.DataFrame,
        cost_model: CostModel,
        action_mapper: ActionMapper,
        feature_stats: FeatureStats,
        feature_columns: list[str] = MOMENTUM_FEATURES,
        risk_per_trade: float = 0.01,
        initial_equity: float = 10_000.0,
        max_holding_bars: int = 120,
    ):
        if risk_per_trade > 0.01:
            raise ValueError("risk_per_trade above 1% violates the project's hard risk cap")
        self.df = df.reset_index(drop=True)
        self.cost = cost_model
        self.mapper = action_mapper
        self.stats = feature_stats
        self.feature_columns = feature_columns
        self.risk_per_trade = risk_per_trade
        self.initial_equity = initial_equity
        self.max_holding_bars = max_holding_bars

        self._features = self.stats.transform(self.df[feature_columns].to_numpy(dtype=float))
        self._open = self.df["open"].to_numpy(dtype=float)
        self._high = self.df["high"].to_numpy(dtype=float)
        self._low = self.df["low"].to_numpy(dtype=float)
        self._close = self.df["close"].to_numpy(dtype=float)
        self._vol = self.df["volatility"].to_numpy(dtype=float)
        self._regime = (
            self.df["volatility_regime"].to_numpy(dtype=object)
            if "volatility_regime" in self.df.columns
            else np.full(len(self.df), None, dtype=object)
        )
        # First bar where every feature is finite (indicators need warmup).
        finite = np.isfinite(self._features).all(axis=1)
        self._first_valid = int(np.argmax(finite)) if finite.any() else len(self.df)

        self.observation_dim = len(feature_columns) + 3
        self.i = 0
        self.equity = initial_equity
        self.position: _EnvPosition | None = None
        self.closed_trades: list[dict] = []

    def reset(self, start: int | None = None, end: int | None = None) -> np.ndarray:
        self._start = max(start if start is not None else self._first_valid, self._first_valid)
        self._end = min(end if end is not None else len(self.df) - 1, len(self.df) - 1)
        self.i = self._start
        self.equity = self.initial_equity
        self.position = None
        self.closed_trades = []
        return self._observe()

    def _observe(self) -> np.ndarray:
        pos_dir, unreal_r, held = 0.0, 0.0, 0.0
        if self.position is not None:
            pos_dir = float(self.position.direction)
            unreal_r = float(
                np.clip(
                    self.position.direction
                    * (self._close[self.i] - self.position.entry_price)
                    / self.position.stop_distance,
                    -3.0,
                    3.0,
                )
            )
            held = (self.i - self.position.entry_index) / 100.0
        obs = np.concatenate([self._features[self.i], [pos_dir, unreal_r, held]])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Advance one bar. The action takes effect at the NEXT bar's open."""
        reward = 0.0
        info: dict = {}
        next_i = self.i + 1
        done = next_i >= self._end

        # 1. Position management at next bar: stop/target/time exits first.
        if self.position is not None:
            exit_raw, reason = self._check_exit(next_i)
            if exit_raw is None and self._wants_close(action):
                exit_raw, reason = self._open[next_i], "signal"
            if exit_raw is None and done:
                exit_raw, reason = self._close[next_i], "end_of_episode"
            if exit_raw is not None:
                reward += self._close_position(next_i, exit_raw, reason, info)
            else:
                reward -= HOLDING_PENALTY

        # 2. Entries execute at next bar's open (only if flat after step 1).
        if self.position is None and self.mapper.direction(action) != 0 and not done:
            self._open_position(action, next_i)

        self.i = next_i
        return self._observe(), float(reward), done, info

    def _wants_close(self, action: int) -> bool:
        direction = self.mapper.direction(action)
        return direction == 0 or (self.position is not None and direction != self.position.direction)

    def _check_exit(self, i: int) -> tuple[float | None, str]:
        pos = self.position
        if pos.direction == 1:
            if self._low[i] <= pos.stop_price:
                return (self._open[i] if self._open[i] < pos.stop_price else pos.stop_price), "stop_loss"
            if self._high[i] >= pos.tp_price:
                return (self._open[i] if self._open[i] > pos.tp_price else pos.tp_price), "take_profit"
        else:
            if self._high[i] >= pos.stop_price:
                return (self._open[i] if self._open[i] > pos.stop_price else pos.stop_price), "stop_loss"
            if self._low[i] <= pos.tp_price:
                return (self._open[i] if self._open[i] < pos.tp_price else pos.tp_price), "take_profit"
        if i - pos.entry_index >= self.max_holding_bars:
            return self._close[i], "time"
        return None, ""

    def _open_position(self, action: int, i: int) -> None:
        direction = self.mapper.direction(action)
        levels = self.mapper.exit_levels(action, self._vol[self.i])
        fill = self.cost.entry_price(self._open[i], direction, self._regime[i])
        stop_distance = fill * levels.stop_loss_pct
        size = (self.equity * self.risk_per_trade / stop_distance) * self.cost.fill_ratio
        size = min(size, self.equity * 0.95 / fill)
        self.position = _EnvPosition(
            direction=direction,
            entry_price=fill,
            size=size,
            stop_price=fill - direction * stop_distance,
            tp_price=fill + direction * fill * levels.take_profit_pct,
            stop_distance=stop_distance,
            entry_fee=self.cost.fee(size * fill),
            entry_index=i,
        )

    def _close_position(self, i: int, exit_raw: float, reason: str, info: dict) -> float:
        pos = self.position
        exit_fill = self.cost.exit_price(exit_raw, pos.direction, self._regime[i])
        fees = pos.entry_fee + self.cost.fee(pos.size * exit_fill)
        pnl = pos.direction * (exit_fill - pos.entry_price) * pos.size - fees
        risk_taken = pos.size * pos.stop_distance
        r_multiple = pnl / risk_taken if risk_taken > 0 else 0.0
        self.equity += pnl
        trade = {
            "pnl": pnl, "r_multiple": r_multiple, "reason": reason,
            "bars_held": i - pos.entry_index, "direction": pos.direction,
        }
        self.closed_trades.append(trade)
        info["closed_trade"] = trade
        self.position = None
        return float(np.clip(r_multiple, -5.0, 5.0))
