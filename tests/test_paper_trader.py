"""End-to-end paper trading loop tests on a replay feed."""
import numpy as np
import pandas as pd
import pytest

from cognition.agents.actions import N_ACTIONS, ActionMapper
from cognition.agents.dqn import DQNAgent, DQNConfig
from cognition.agents.ensemble import EnsembleMember, EnsembleStrategy
from cognition.agents.env import FeatureStats
from cognition.agents.specs import AGENT_SPECS
from cognition.backtest.costs import CostModel
from cognition.backtest.simulator import Signal
from cognition.backtest.synthetic import make_synthetic_ohlcv
from cognition.data.feed import ReplayFeed
from cognition.features.engineering import extract_features
from cognition.learning.journal import TradeJournal
from cognition.monitoring.alerts import TelegramAlerter
from cognition.monitoring.dashboard import DashboardWriter
from cognition.paper.trader import PaperTrader
from cognition.risk.engine import RiskEngine, RiskLimits


class AlwaysLongStrategy:
    """Deterministic stand-in: goes long on every bar it can."""

    def __init__(self, stop=0.01, tp=0.02, max_holding=20):
        self.signal_args = Signal(direction=1, stop_loss_pct=stop, take_profit_pct=tp, max_holding_bars=max_holding)
        self.closed = []

    def fit(self, df):
        pass

    def prepare(self, df):
        pass

    def signal(self, i):
        return self.signal_args

    def on_trade_closed(self, trade):
        self.closed.append(trade)

    def last_votes_snapshot(self):
        return {"scripted": {"direction": 1, "confidence": 1.0}}


def make_trader(tmp_path, strategy, df, symbol="BTC/USDT", **overrides):
    journal = TradeJournal(tmp_path / "journal.db")
    alerts_sent = []
    alerter = TelegramAlerter(token="t", chat_id="c", transport=lambda *a: alerts_sent.append(a), clock=lambda: 0.0)
    dashboard = DashboardWriter(tmp_path / "dash.html")
    defaults = dict(
        symbols=[symbol],
        strategies={symbol: strategy},
        risk_engine=RiskEngine(RiskLimits(max_trades_per_hour=1000, max_trades_per_day=10_000)),
        cost_model=CostModel(),
        journal=journal,
        alerter=alerter,
        dashboard=dashboard,
        initial_equity=10_000.0,
        warmup_bars=60,
        buffer_bars=120,
        dashboard_every=50,
        regime_window_bars=30,
    )
    defaults.update(overrides)
    trader = PaperTrader(**defaults)
    trader._alerts_sent = alerts_sent
    return trader, journal


@pytest.fixture()
def market_df():
    return make_synthetic_ohlcv(n_bars=400, seed=21)


def test_loop_trades_journals_and_updates_equity(tmp_path, market_df):
    strategy = AlwaysLongStrategy()
    trader, journal = make_trader(tmp_path, strategy, market_df)
    trader.run(ReplayFeed({"BTC/USDT": market_df}))

    trades = journal.trades()
    assert len(trades) > 0
    assert trader.equity == pytest.approx(10_000.0 + trades["pnl"].sum())
    # Learning-loop hook fired for every close.
    assert len(strategy.closed) == len(trades)
    # Journal captured votes + feature snapshot.
    first = trades.iloc[0]
    assert first["agent_votes"]["scripted"]["confidence"] == 1.0
    assert "rsi_14" in first["feature_snapshot"]
    assert first["market_regime"] is not None
    # Equity curve recorded.
    assert len(journal.equity_curve()) > 0


def test_no_trades_before_warmup(tmp_path, market_df):
    strategy = AlwaysLongStrategy()
    trader, journal = make_trader(tmp_path, strategy, market_df, warmup_bars=390)
    trader.run(ReplayFeed({"BTC/USDT": market_df}))
    # Only ~10 tradeable bars after warmup; entries possible but none
    # before the buffer filled: entry timestamps must be after warmup.
    trades = journal.trades()
    if len(trades):
        first_tradeable_ts = market_df["timestamp"].iloc[389]  # bar that completes the warmup
        assert (trades["entry_ts"] >= first_tradeable_ts).all()


def test_stops_and_costs_are_applied(tmp_path, market_df):
    strategy = AlwaysLongStrategy(stop=0.002, tp=0.004)
    trader, journal = make_trader(tmp_path, strategy, market_df)
    trader.run(ReplayFeed({"BTC/USDT": market_df}))
    trades = journal.trades()
    assert len(trades) > 3
    assert (trades["fees"] > 0).all()
    assert set(trades["exit_reason"]) <= {"stop_loss", "take_profit", "time"}
    # A stop-out should cost about 1R + cost drag, never less than -1R.
    stops = trades[trades["exit_reason"] == "stop_loss"]
    if len(stops):
        assert (stops["r_multiple"] <= -0.99).all()
        assert (stops["r_multiple"] > -4.0).all()


def test_circuit_breaker_halts_trading_and_alerts(tmp_path):
    # Crash the market so longs bleed until the daily breaker trips.
    n = 400
    rng = np.random.default_rng(5)
    drift = np.full(n, -0.004)
    close = 50_000 * np.cumprod(1 + drift + rng.normal(0, 0.0005, n))
    open_ = np.concatenate([[50_000], close[:-1]])
    df = pd.DataFrame({
        "timestamp": 1_700_006_400_000 + np.arange(n) * 60_000,
        "open": open_, "high": np.maximum(open_, close) * 1.0001,
        "low": np.minimum(open_, close) * 0.9999, "close": close,
        "volume": 5.0,
    })
    strategy = AlwaysLongStrategy(stop=0.005, tp=0.05, max_holding=50)
    trader, journal = make_trader(
        tmp_path, strategy, df,
        risk_engine=RiskEngine(RiskLimits(max_trades_per_hour=1000, max_trades_per_day=10_000)),
    )
    trader.run(ReplayFeed({"BTC/USDT": df}))

    assert trader.risk.is_halted, "breaker should have tripped in a -40% day"
    assert any("circuit_breaker" in str(a) for a in trader._alerts_sent)
    events = journal.events(limit=50)
    assert (events["kind"] == "circuit_breaker").any()
    # No entries after the halt: every entry precedes the halt event time.
    halt_ts = int(events[events["kind"] == "circuit_breaker"]["ts"].iloc[0])
    trades = journal.trades()
    assert (trades["entry_ts"] <= halt_ts).all()


def test_dashboard_file_written(tmp_path, market_df):
    strategy = AlwaysLongStrategy()
    trader, _ = make_trader(tmp_path, strategy, market_df, dashboard_every=10)
    trader.run(ReplayFeed({"BTC/USDT": market_df}))
    html = (tmp_path / "dash.html").read_text()
    assert "Project Cognition" in html


def test_ensemble_online_learning_updates_agents_in_loop(tmp_path, market_df):
    featured = extract_features(make_synthetic_ohlcv(n_bars=1500, seed=3))
    members = []
    for k, spec in enumerate(AGENT_SPECS.values()):
        stats = FeatureStats.fit(featured, spec.feature_columns)
        agent = DQNAgent(
            len(spec.feature_columns) + 3, N_ACTIONS,
            DQNConfig(seed=k, warmup_steps=2, batch_size=2),
        )
        members.append(EnsembleMember(spec.name, agent, stats, spec.feature_columns))
    strategy = EnsembleStrategy(members, ActionMapper(), vote_threshold=0.01, online_learning=True)

    trader, journal = make_trader(tmp_path, strategy, market_df)
    trader.run(ReplayFeed({"BTC/USDT": market_df}))

    closed = journal.trade_count()
    if closed:  # untrained ensembles usually trade at this low threshold
        assert strategy.meta.updates == closed
        # Per-trade transitions reached the agents' replay buffers.
        assert all(len(m.agent.buffer) == closed for m in members)
