"""Paper trading loop (Milestone 6).

Drives the full production pipeline off a live (or replayed) bar feed:

    closed bar -> features -> regime label -> ensemble vote
              -> Guardian A (Risk Engine) -> paper fill (cost model)
              -> position management (stop/target/time, conservative)
              -> on close: journal + meta-learner + per-trade agent update
              -> Guardian B: alerts, dashboard, watchdog, equity log

Fills are simulated with the SAME CostModel as the backtester (taker fee
+ regime-aware adverse slippage) — deliberately pessimistic versus the
limit-first executor we'll use live, so paper results understate rather
than flatter. The identical loop runs on a ReplayFeed (sandbox
verification) and live feeds (REST polling / WebSocket).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cognition.backtest.costs import CostModel
from cognition.backtest.regimes import label_market_regimes
from cognition.backtest.simulator import Trade
from cognition.data.feed import BarEvent
from cognition.features.engineering import extract_features
from cognition.learning.journal import TradeJournal
from cognition.monitoring.alerts import FeedWatchdog, TelegramAlerter
from cognition.monitoring.dashboard import DashboardWriter
from cognition.risk.engine import RiskEngine, TradeProposal
from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("paper_trader")

FEATURE_SNAPSHOT_COLUMNS = [
    "price_velocity", "price_acceleration", "rsi_14", "macd_diff",
    "bollinger_position", "volume_zscore", "sma21_distance",
    "donchian_position", "range_pct", "close_in_range", "volatility",
]


@dataclass
class PaperPosition:
    symbol: str
    direction: int
    entry_ts: int
    entry_bar: int
    ideal_price: float
    entry_price: float
    size: float
    notional: float
    stop_price: float
    tp_price: float
    stop_distance: float
    entry_fee: float
    max_holding_bars: int | None
    worst_price: float
    market_regime: object
    session: object
    agent_votes: dict = field(default_factory=dict)
    feature_snapshot: dict = field(default_factory=dict)


@dataclass
class _SymbolState:
    buffer: list[dict] = field(default_factory=list)
    position: PaperPosition | None = None
    bars_seen: int = 0


class PaperTrader:
    def __init__(
        self,
        symbols: list[str],
        strategies: dict[str, object],          # symbol -> Strategy-like
        risk_engine: RiskEngine,
        cost_model: CostModel,
        journal: TradeJournal,
        alerter: TelegramAlerter,
        dashboard: DashboardWriter | None = None,
        initial_equity: float = 10_000.0,
        warmup_bars: int = 250,
        buffer_bars: int = 600,
        dashboard_every: int = 20,
        regime_window_bars: int = 300,
        regime_threshold: float = 0.03,
        timeframe_ms: int = 60_000,
        watchdog_stale_multiple: float = 3.0,
        realtime: bool = False,
    ):
        self.symbols = symbols
        self.strategies = strategies
        self.risk = risk_engine
        self.cost = cost_model
        self.journal = journal
        self.alerter = alerter
        self.dashboard = dashboard
        self.equity = initial_equity
        self.initial_equity = initial_equity
        self.warmup_bars = warmup_bars
        self.buffer_bars = max(buffer_bars, warmup_bars + 10)
        self.dashboard_every = dashboard_every
        self.regime_window_bars = regime_window_bars
        self.regime_threshold = regime_threshold
        self.realtime = realtime
        self.watchdog = FeedWatchdog(timeframe_ms, watchdog_stale_multiple)

        self._state: dict[str, _SymbolState] = {s: _SymbolState() for s in symbols}
        self._bars_processed = 0
        self._was_halted = False
        self._equity_history: list[float] = [initial_equity]
        self._wins = 0
        self._closed = 0
        self._current_regime: object = "unknown"

    # ---- main loop ------------------------------------------------------

    def run(self, feed, max_bars: int | None = None) -> None:
        log_with_fields(
            logger, 20, "Paper trading started",
            symbols=self.symbols, equity=self.equity, realtime=self.realtime,
        )
        for event in feed.bars():
            self.on_bar(event)
            if max_bars is not None and self._bars_processed >= max_bars:
                break
        log_with_fields(
            logger, 20, "Paper trading stopped",
            bars=self._bars_processed, trades=self._closed, equity=round(self.equity, 2),
        )

    def on_bar(self, event: BarEvent) -> None:
        state = self._state.get(event.symbol)
        if state is None:
            return
        self._bars_processed += 1
        state.bars_seen += 1
        self.watchdog.beat(self._now_ms(event))
        state.buffer.append(event.__dict__ | {})
        if len(state.buffer) > self.buffer_bars:
            state.buffer = state.buffer[-self.buffer_bars:]
        if len(state.buffer) < self.warmup_bars:
            return

        df = self._featured_frame(state)
        i = len(df) - 1
        row = df.iloc[i]
        self._current_regime = row["market_regime"]

        strategy = self.strategies[event.symbol]
        strategy.prepare(df)

        if state.position is not None:
            self._manage_position(state, event, row)
        if state.position is None and not self.risk.is_halted:
            self._maybe_enter(state, event, row, strategy, df, i)

        self._check_halt_transition(event)
        mark = self._mark_to_market(event)
        self._equity_history.append(mark)
        self.journal.record_equity(event.timestamp, mark)
        if self.dashboard and self._bars_processed % self.dashboard_every == 0:
            self._write_dashboard()

    # ---- pipeline steps -------------------------------------------------

    def _featured_frame(self, state: _SymbolState) -> pd.DataFrame:
        df = pd.DataFrame(state.buffer)
        df = extract_features(df)
        window = min(self.regime_window_bars, max(len(df) // 3, 10))
        return label_market_regimes(df, window_bars=window, threshold=self.regime_threshold)

    def _maybe_enter(self, state, event: BarEvent, row, strategy, df: pd.DataFrame, i: int) -> None:
        signal = strategy.signal(i)
        if signal is None or signal.direction == 0:
            return
        proposal = TradeProposal(
            symbol=event.symbol, direction=signal.direction, price=event.close,
            stop_loss_pct=signal.stop_loss_pct, take_profit_pct=signal.take_profit_pct,
            timestamp_ms=event.timestamp,
        )
        decision = self.risk.evaluate(proposal, self.equity)
        if not decision.approved:
            self.journal.record_event(event.timestamp, "warning", "risk_rejection", decision.reason)
            return

        regime = row.get("volatility_regime")
        fill = self.cost.entry_price(event.close, signal.direction, regime)
        stop_distance = fill * decision.stop_loss_pct
        size = decision.quantity
        entry_fee = self.cost.fee(size * fill)
        votes = strategy.last_votes_snapshot() if hasattr(strategy, "last_votes_snapshot") else {}
        snapshot = {
            c: (None if pd.isna(row[c]) else float(row[c]))
            for c in FEATURE_SNAPSHOT_COLUMNS if c in row.index
        }
        state.position = PaperPosition(
            symbol=event.symbol, direction=signal.direction,
            entry_ts=event.timestamp, entry_bar=state.bars_seen,
            ideal_price=event.close, entry_price=fill,
            size=size, notional=size * fill,
            stop_price=fill - signal.direction * stop_distance,
            tp_price=fill + signal.direction * fill * decision.take_profit_pct,
            stop_distance=stop_distance, entry_fee=entry_fee,
            max_holding_bars=signal.max_holding_bars,
            worst_price=fill,
            market_regime=row["market_regime"], session=row.get("session"),
            agent_votes=votes, feature_snapshot=snapshot,
        )
        self.risk.record_open(event.symbol, state.position.notional, event.timestamp)
        log_with_fields(
            logger, 20, "Paper position opened",
            symbol=event.symbol, direction=signal.direction, size=round(size, 8),
            entry=round(fill, 4), stop=round(state.position.stop_price, 4),
            tp=round(state.position.tp_price, 4), votes=votes,
        )

    def _manage_position(self, state: _SymbolState, event: BarEvent, row) -> None:
        pos = state.position
        exit_raw, reason = None, ""
        if pos.direction == 1:
            pos.worst_price = min(pos.worst_price, event.low)
            if event.low <= pos.stop_price:
                exit_raw = event.open if event.open < pos.stop_price else pos.stop_price
                reason = "stop_loss"
            elif pos.tp_price > 0 and event.high >= pos.tp_price:
                exit_raw = event.open if event.open > pos.tp_price else pos.tp_price
                reason = "take_profit"
        else:
            pos.worst_price = max(pos.worst_price, event.high)
            if event.high >= pos.stop_price:
                exit_raw = event.open if event.open > pos.stop_price else pos.stop_price
                reason = "stop_loss"
            elif pos.tp_price > 0 and event.low <= pos.tp_price:
                exit_raw = event.open if event.open < pos.tp_price else pos.tp_price
                reason = "take_profit"
        held = state.bars_seen - pos.entry_bar
        if exit_raw is None and pos.max_holding_bars is not None and held >= pos.max_holding_bars:
            exit_raw, reason = event.close, "time"
        if exit_raw is None:
            return

        regime = row.get("volatility_regime")
        exit_fill = self.cost.exit_price(exit_raw, pos.direction, regime)
        fees = pos.entry_fee + self.cost.fee(pos.size * exit_fill)
        pnl = pos.direction * (exit_fill - pos.entry_price) * pos.size - fees
        risk_taken = pos.size * pos.stop_distance
        r_multiple = pnl / risk_taken if risk_taken > 0 else 0.0
        slippage = (abs(pos.entry_price - pos.ideal_price) + abs(exit_fill - exit_raw)) * pos.size
        mae = pos.direction * (pos.entry_price - pos.worst_price)
        mae_r = max(mae, 0.0) / pos.stop_distance if pos.stop_distance > 0 else 0.0

        self.equity += pnl
        self._closed += 1
        self._wins += int(pnl > 0)
        self.risk.record_close(pos.symbol, self.equity)

        trade = Trade(
            direction=pos.direction, entry_timestamp=pos.entry_ts, exit_timestamp=event.timestamp,
            entry_price=pos.entry_price, exit_price=exit_fill, size=pos.size,
            pnl=pnl, r_multiple=r_multiple, fees=fees, slippage_cost=slippage,
            exit_reason=reason, bars_held=held, mae_r=mae_r,
            market_regime=pos.market_regime, session=pos.session,
        )
        strategy = self.strategies[pos.symbol]
        hook = getattr(strategy, "on_trade_closed", None)
        if hook is not None:
            hook(trade)  # meta-learner reweights + per-trade agent updates

        self.journal.record_trade(
            symbol=pos.symbol, direction=pos.direction,
            entry_ts=pos.entry_ts, exit_ts=event.timestamp,
            entry_price=pos.entry_price, exit_price=exit_fill, size=pos.size,
            pnl=pnl, r_multiple=r_multiple, fees=fees, slippage=slippage,
            exit_reason=reason, bars_held=held, mae_r=mae_r,
            market_regime=pos.market_regime, session=pos.session,
            agent_votes=pos.agent_votes, feature_snapshot=pos.feature_snapshot,
            equity_after=self.equity,
        )
        log_with_fields(
            logger, 20, "Paper position closed",
            symbol=pos.symbol, reason=reason, pnl=round(pnl, 4),
            r=round(r_multiple, 3), equity=round(self.equity, 2),
        )
        state.position = None

    # ---- guardians ------------------------------------------------------

    def _check_halt_transition(self, event: BarEvent) -> None:
        if self.risk.is_halted and not self._was_halted:
            message = f"Daily circuit breaker tripped — trading halted. Equity: {self.equity:.2f}"
            self.alerter.alert("circuit_breaker", message, critical=True)
            self.journal.record_event(event.timestamp, "critical", "circuit_breaker", message)
        self._was_halted = self.risk.is_halted

    def check_feed_health(self, now_ms: int | None = None) -> bool:
        """Call periodically in realtime mode (the CLI runs this on a
        timer thread). Returns True if the feed looks healthy."""
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        if self.watchdog.is_stale(now_ms):
            self.alerter.alert("feed_stale", "No market data received — feed stalled or exchange down.", critical=True)
            return False
        return True

    def _now_ms(self, event: BarEvent) -> int:
        return int(time.time() * 1000) if self.realtime else event.timestamp

    def _mark_to_market(self, event: BarEvent) -> float:
        unrealized = 0.0
        for state in self._state.values():
            pos = state.position
            if pos is not None and pos.symbol == event.symbol:
                unrealized += pos.direction * (event.close - pos.entry_price) * pos.size
        return self.equity + unrealized

    # ---- dashboard ------------------------------------------------------

    def _write_dashboard(self) -> None:
        positions = pd.DataFrame([
            {
                "symbol": p.symbol, "direction": p.direction, "size": p.size,
                "entry": p.entry_price, "stop": p.stop_price, "target": p.tp_price,
            }
            for s in self._state.values() if (p := s.position) is not None
        ])
        weights = pd.DataFrame()
        session_table = pd.DataFrame()
        for strategy in self.strategies.values():
            if hasattr(strategy, "meta"):
                weights = strategy.meta.snapshot()
            if hasattr(strategy, "session_tracker"):
                session_table = strategy.session_tracker.table()
        self.dashboard.write(
            equity_values=self._equity_history,
            initial_equity=self.initial_equity,
            closed_trades=self._closed,
            win_rate=self._wins / self._closed if self._closed else 0.0,
            current_regime=self._current_regime,
            halted=self.risk.is_halted,
            positions=positions,
            weights=weights,
            session_table=session_table,
            recent_trades=self.journal.trades(limit=15).drop(
                columns=["agent_votes", "feature_snapshot"], errors="ignore"
            ),
        )
