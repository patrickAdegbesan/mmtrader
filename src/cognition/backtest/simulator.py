"""Event-driven, per-bar backtest engine with the CostModel applied to
every fill.

Design decisions that protect against the classic backtest lies:
- No lookahead: a signal computed at the close of bar i can execute no
  earlier than the open of bar i+latency_bars.
- Conservative intra-bar ordering: if a bar touches both the stop-loss
  and the take-profit, the stop-loss is assumed to have hit first.
- Gaps: if a bar opens beyond the stop, the fill is the (worse) open
  price, not the stop price.
- Every entry must carry a stop-loss; signals without one are refused.
  Position size is derived from the stop distance so each trade risks a
  fixed fraction of current equity (risk-based sizing, the same rule the
  Risk Engine will enforce for real in Milestone 5).

This same engine doubles as the RL environment substrate in Milestone 3:
an agent is just a Strategy whose signal() consults a policy network.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from cognition.backtest.costs import CostModel


@dataclass(frozen=True)
class Signal:
    direction: int                      # 1 long, -1 short, 0 close-position
    stop_loss_pct: float = 0.0          # distance from entry, e.g. 0.004 = 0.4%
    take_profit_pct: float = 0.0
    max_holding_bars: int | None = None


class Strategy(Protocol):
    def fit(self, train_df: pd.DataFrame) -> None: ...
    def prepare(self, df: pd.DataFrame) -> None: ...
    def signal(self, i: int) -> Signal | None: ...


@dataclass
class Trade:
    direction: int
    entry_timestamp: int
    exit_timestamp: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    r_multiple: float
    fees: float
    slippage_cost: float
    exit_reason: str
    bars_held: int
    mae_r: float                        # max adverse excursion, in R
    market_regime: object = None
    session: object = None


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    initial_equity: float
    final_equity: float

    @property
    def total_fees(self) -> float:
        return sum(t.fees for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage_cost for t in self.trades)


@dataclass
class _Position:
    direction: int
    entry_index: int
    entry_price: float
    ideal_entry_price: float
    size: float
    stop_price: float
    tp_price: float
    stop_distance: float
    entry_fee: float
    max_holding_bars: int | None
    worst_price: float                  # most adverse price seen while held


class EventDrivenBacktester:
    def __init__(
        self,
        cost_model: CostModel,
        initial_equity: float = 10_000.0,
        risk_per_trade: float = 0.01,
        max_position_fraction: float = 0.95,
    ):
        if risk_per_trade > 0.01:
            raise ValueError("risk_per_trade above 1% violates the project's hard risk cap")
        self.cost = cost_model
        self.initial_equity = initial_equity
        self.risk_per_trade = risk_per_trade
        self.max_position_fraction = max_position_fraction

    def run(self, df: pd.DataFrame, strategy: Strategy, start_index: int = 0) -> BacktestResult:
        """Run over `df` (must be sorted, default-indexed OHLCV+features).
        `start_index` lets walk-forward callers include warmup bars for
        indicator computation without allowing trades in them.
        """
        n = len(df)
        ts = df["timestamp"].to_numpy()
        open_ = df["open"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        vol_regime = (
            df["volatility_regime"].to_numpy(dtype=object)
            if "volatility_regime" in df.columns else np.full(n, None, dtype=object)
        )
        market_regime = (
            df["market_regime"].to_numpy(dtype=object)
            if "market_regime" in df.columns else np.full(n, None, dtype=object)
        )
        session = (
            df["session"].to_numpy(dtype=object)
            if "session" in df.columns else np.full(n, None, dtype=object)
        )

        strategy.prepare(df)

        equity = self.initial_equity
        curve = np.empty(n, dtype=float)
        trades: list[Trade] = []
        pos: _Position | None = None
        pending_entry: tuple[int, Signal] | None = None   # (execute_at, signal)
        pending_exit_at: int | None = None

        for i in range(n):
            # 1. Execute a scheduled entry at this bar's open.
            if pending_entry is not None and i >= pending_entry[0]:
                if pos is None:
                    pos = self._open_position(pending_entry[1], i, open_[i], vol_regime[i], equity)
                pending_entry = None

            # 2. Manage the open position: scheduled exit, stop, target, time.
            if pos is not None:
                exit_raw, reason = None, ""
                if pending_exit_at is not None and i >= pending_exit_at and i > pos.entry_index:
                    exit_raw, reason = open_[i], "signal"
                elif pos.direction == 1:
                    pos.worst_price = min(pos.worst_price, low[i])
                    if low[i] <= pos.stop_price:
                        exit_raw = open_[i] if open_[i] < pos.stop_price else pos.stop_price
                        reason = "stop_loss"
                    elif pos.tp_price > 0 and high[i] >= pos.tp_price:
                        exit_raw = open_[i] if open_[i] > pos.tp_price else pos.tp_price
                        reason = "take_profit"
                else:
                    pos.worst_price = max(pos.worst_price, high[i])
                    if high[i] >= pos.stop_price:
                        exit_raw = open_[i] if open_[i] > pos.stop_price else pos.stop_price
                        reason = "stop_loss"
                    elif pos.tp_price > 0 and low[i] <= pos.tp_price:
                        exit_raw = open_[i] if open_[i] < pos.tp_price else pos.tp_price
                        reason = "take_profit"
                if exit_raw is None and pos.max_holding_bars is not None and i - pos.entry_index >= pos.max_holding_bars:
                    exit_raw, reason = close[i], "time"

                if exit_raw is not None:
                    trade, pnl = self._close_position(pos, i, exit_raw, reason, ts, vol_regime, market_regime, session)
                    equity += pnl
                    trades.append(trade)
                    pos = None
                    pending_exit_at = None
                    self._notify_close(strategy, trade)

            # 3. Consult the strategy at this bar's close (trades allowed only past start_index).
            if i >= start_index:
                sig = strategy.signal(i)
                if sig is not None:
                    if pos is None and pending_entry is None and sig.direction in (1, -1):
                        if sig.stop_loss_pct > 0 and i + self.cost.latency_bars < n:
                            pending_entry = (i + self.cost.latency_bars, sig)
                    elif pos is not None and sig.direction != pos.direction and pending_exit_at is None:
                        pending_exit_at = i + self.cost.latency_bars

            # 4. Mark-to-market equity curve.
            unrealized = pos.direction * (close[i] - pos.entry_price) * pos.size if pos is not None else 0.0
            curve[i] = equity + unrealized

        # Force-close anything still open at the last bar's close.
        if pos is not None:
            trade, pnl = self._close_position(pos, n - 1, close[-1], "end_of_data", ts, vol_regime, market_regime, session)
            equity += pnl
            trades.append(trade)
            curve[-1] = equity
            self._notify_close(strategy, trade)

        equity_curve = pd.Series(curve, index=pd.to_datetime(ts, unit="ms", utc=True), name="equity")
        return BacktestResult(trades=trades, equity_curve=equity_curve, initial_equity=self.initial_equity, final_equity=equity)

    @staticmethod
    def _notify_close(strategy: Strategy, trade: Trade) -> None:
        """Learning-loop hook: strategies (e.g. the ensemble's meta-learner)
        may update themselves after every closed trade."""
        hook = getattr(strategy, "on_trade_closed", None)
        if hook is not None:
            hook(trade)

    def _open_position(self, sig: Signal, i: int, bar_open: float, regime: object, equity: float) -> _Position:
        fill = self.cost.entry_price(bar_open, sig.direction, regime)
        stop_distance = fill * sig.stop_loss_pct
        risk_amount = equity * self.risk_per_trade
        size = (risk_amount / stop_distance) * self.cost.fill_ratio
        size = min(size, equity * self.max_position_fraction / fill)

        stop_price = fill - sig.direction * stop_distance
        tp_price = fill + sig.direction * fill * sig.take_profit_pct if sig.take_profit_pct > 0 else 0.0
        entry_fee = self.cost.fee(size * fill)

        return _Position(
            direction=sig.direction,
            entry_index=i,
            entry_price=fill,
            ideal_entry_price=bar_open,
            size=size,
            stop_price=stop_price,
            tp_price=tp_price,
            stop_distance=stop_distance,
            entry_fee=entry_fee,
            max_holding_bars=sig.max_holding_bars,
            worst_price=fill,
        )

    def _close_position(
        self,
        pos: _Position,
        i: int,
        exit_raw: float,
        reason: str,
        ts: np.ndarray,
        vol_regime: np.ndarray,
        market_regime: np.ndarray,
        session: np.ndarray,
    ) -> tuple[Trade, float]:
        exit_fill = self.cost.exit_price(exit_raw, pos.direction, vol_regime[i])
        exit_fee = self.cost.fee(pos.size * exit_fill)
        fees = pos.entry_fee + exit_fee
        pnl = pos.direction * (exit_fill - pos.entry_price) * pos.size - fees

        risk_taken = pos.size * pos.stop_distance
        r_multiple = pnl / risk_taken if risk_taken > 0 else 0.0
        slippage_cost = (
            abs(pos.entry_price - pos.ideal_entry_price) + abs(exit_fill - exit_raw)
        ) * pos.size
        mae = pos.direction * (pos.entry_price - pos.worst_price)
        mae_r = max(mae, 0.0) / pos.stop_distance if pos.stop_distance > 0 else 0.0

        trade = Trade(
            direction=pos.direction,
            entry_timestamp=int(ts[pos.entry_index]),
            exit_timestamp=int(ts[i]),
            entry_price=pos.entry_price,
            exit_price=exit_fill,
            size=pos.size,
            pnl=pnl,
            r_multiple=r_multiple,
            fees=fees,
            slippage_cost=slippage_cost,
            exit_reason=reason,
            bars_held=i - pos.entry_index,
            mae_r=mae_r,
            market_regime=market_regime[pos.entry_index],
            session=session[pos.entry_index],
        )
        return trade, pnl
