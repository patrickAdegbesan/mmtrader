"""Tests for the meta-learner (regime memory, online updates), session
tracking, vote combination, and the ensemble running end-to-end through
the backtester with the close-hook firing."""
import numpy as np
import pandas as pd
import pytest

from cognition.agents.actions import N_ACTIONS, ActionMapper
from cognition.agents.dqn import DQNAgent, DQNConfig
from cognition.agents.ensemble import EnsembleMember, EnsembleStrategy
from cognition.agents.env import FeatureStats
from cognition.agents.meta import AgentVote, MetaLearner, SessionPerformanceTracker
from cognition.agents.specs import AGENT_NAMES, AGENT_SPECS
from cognition.backtest.costs import CostModel
from cognition.backtest.regimes import label_market_regimes
from cognition.backtest.simulator import EventDrivenBacktester, Signal, Trade
from cognition.backtest.strategies import ScriptedStrategy
from cognition.backtest.synthetic import make_synthetic_ohlcv
from cognition.features.engineering import extract_features


# ---------- meta-learner ----------

def vote(direction, confidence=1.0):
    return AgentVote(direction=direction, confidence=confidence)


def test_weights_start_uniform_and_sum_to_one():
    meta = MetaLearner(["a", "b", "c", "d"])
    w = meta.weights("bull")
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(0.25) for v in w.values())


def test_agent_agreeing_with_winners_gains_weight():
    meta = MetaLearner(["a", "b"])
    for _ in range(10):
        # 'a' votes with the winning direction, 'b' against it.
        meta.update({"a": vote(1), "b": vote(-1)}, executed_direction=1, r_multiple=2.0, regime="bull")
    w = meta.weights("bull")
    assert w["a"] > 0.5 > w["b"]
    assert meta.updates == 10


def test_agent_agreeing_with_losers_loses_weight():
    meta = MetaLearner(["a", "b"])
    for _ in range(10):
        meta.update({"a": vote(1), "b": vote(0)}, executed_direction=1, r_multiple=-1.5, regime="bear")
    w = meta.weights("bear")
    assert w["a"] < w["b"]  # 'a' backed losers; abstaining 'b' is now preferred


def test_regime_memory_keeps_weights_separate_per_regime():
    meta = MetaLearner(["a", "b"])
    for _ in range(10):
        meta.update({"a": vote(1), "b": vote(-1)}, executed_direction=1, r_multiple=2.0, regime="bull")
        meta.update({"a": vote(1), "b": vote(-1)}, executed_direction=1, r_multiple=-2.0, regime="sideways")
    assert meta.weights("bull")["a"] > 0.5          # a is good in bull...
    assert meta.weights("sideways")["a"] < 0.5      # ...and bad in sideways
    assert meta.weights("never_seen")["a"] == pytest.approx(0.5)  # unseen regime: uniform


def test_confidence_scales_the_update():
    meta_hi = MetaLearner(["a", "b"])
    meta_lo = MetaLearner(["a", "b"])
    for _ in range(5):
        meta_hi.update({"a": vote(1, 1.0), "b": vote(0)}, 1, 2.0, "bull")
        meta_lo.update({"a": vote(1, 0.1), "b": vote(0)}, 1, 2.0, "bull")
    assert meta_hi.weights("bull")["a"] > meta_lo.weights("bull")["a"]


def test_snapshot_lists_seen_regimes():
    meta = MetaLearner(["a", "b"])
    meta.update({"a": vote(1), "b": vote(0)}, 1, 1.0, "bull")
    meta.update({"a": vote(1), "b": vote(0)}, 1, 1.0, "bear")
    snap = meta.snapshot()
    assert set(snap.index) == {"bull", "bear"}
    np.testing.assert_allclose(snap.sum(axis=1), 1.0)


# ---------- session tracker ----------

def test_session_tracker_accumulates_per_cell():
    tracker = SessionPerformanceTracker()
    tracker.update("bull", "asian", pnl=10.0, r_multiple=1.0)
    tracker.update("bull", "asian", pnl=-5.0, r_multiple=-0.5)
    tracker.update("bear", "us", pnl=3.0, r_multiple=0.3)
    table = tracker.table()
    assert len(table) == 2
    bull_asian = table[(table["regime"] == "bull") & (table["session"] == "asian")].iloc[0]
    assert bull_asian["trades"] == 2
    assert bull_asian["win_rate"] == pytest.approx(0.5)
    assert bull_asian["total_pnl"] == pytest.approx(5.0)


# ---------- engine close-hook ----------

def test_engine_calls_on_trade_closed_hook():
    df = pd.DataFrame({
        "timestamp": [1_700_000_000_000 + i * 60_000 for i in range(20)],
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0,
    })
    closed = []

    class HookedStrategy(ScriptedStrategy):
        def on_trade_closed(self, trade):
            closed.append(trade)

    strat = HookedStrategy({2: Signal(direction=1, stop_loss_pct=0.05, max_holding_bars=3)})
    EventDrivenBacktester(CostModel()).run(df, strat)
    assert len(closed) == 1
    assert isinstance(closed[0], Trade)


# ---------- ensemble end-to-end ----------

@pytest.fixture(scope="module")
def ensemble_setup():
    df = make_synthetic_ohlcv(n_bars=4000, seed=11)
    df = extract_features(df)
    df = label_market_regimes(df, window_bars=400)
    members = []
    for k, spec in enumerate(AGENT_SPECS.values()):
        stats = FeatureStats.fit(df, spec.feature_columns)
        agent = DQNAgent(len(spec.feature_columns) + 3, N_ACTIONS, DQNConfig(seed=k))
        members.append(EnsembleMember(spec.name, agent, stats, spec.feature_columns))
    return df, members


def test_extended_feature_columns_exist(ensemble_setup):
    df, _ = ensemble_setup
    for spec in AGENT_SPECS.values():
        for col in spec.feature_columns:
            assert col in df.columns, f"{spec.name} needs missing column {col}"


def test_ensemble_votes_and_logs_decisions(ensemble_setup):
    df, members = ensemble_setup
    strategy = EnsembleStrategy(members, ActionMapper(), vote_threshold=0.0)
    strategy.prepare(df)
    sig = None
    for i in range(100, 300):
        sig = strategy.signal(i)
        if sig is not None:
            break
    assert strategy.decision_log, "no decisions were logged"
    entry = strategy.decision_log[-1]
    assert set(entry["votes"]) == set(AGENT_NAMES)
    assert sum(entry["weights"].values()) == pytest.approx(1.0)
    if sig is not None:
        assert sig.stop_loss_pct > 0


def test_high_threshold_blocks_all_trades(ensemble_setup):
    df, members = ensemble_setup
    strategy = EnsembleStrategy(members, ActionMapper(), vote_threshold=10.0)
    strategy.prepare(df)
    assert all(strategy.signal(i) is None for i in range(100, 400))


def test_ensemble_through_backtester_updates_meta_online(ensemble_setup):
    df, members = ensemble_setup
    meta = MetaLearner([m.name for m in members])
    strategy = EnsembleStrategy(members, ActionMapper(), meta=meta, vote_threshold=0.05)
    engine = EventDrivenBacktester(CostModel(), initial_equity=10_000)
    result = engine.run(df, strategy)
    assert result.final_equity > 0
    if result.trades:  # untrained agents usually trade at threshold 0.05
        assert meta.updates == len(result.trades)
        assert not strategy.session_tracker.table().empty
        assert not meta.snapshot().empty
