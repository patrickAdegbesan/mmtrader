"""Tests for the RL stack: actions, env mechanics, network, replay,
DQN agent, and the backtester adapter."""
import numpy as np
import pandas as pd
import pytest
import torch

from cognition.agents.actions import (
    FLAT, LONG_TIGHT, LONG_WIDE, N_ACTIONS, SHORT_TIGHT, ActionMapper,
)
from cognition.agents.dqn import DQNAgent, DQNConfig
from cognition.agents.env import MOMENTUM_FEATURES, FeatureStats, ScalpingEnv
from cognition.agents.momentum import DQNStrategy
from cognition.agents.networks import DuelingQNetwork
from cognition.agents.replay import ReplayBuffer
from cognition.backtest.costs import CostModel
from cognition.backtest.simulator import EventDrivenBacktester
from cognition.backtest.synthetic import make_synthetic_ohlcv
from cognition.features.engineering import extract_features


@pytest.fixture(scope="module")
def featured_df():
    df = make_synthetic_ohlcv(n_bars=3000, seed=9)
    return extract_features(df)


def make_env(df, **kwargs):
    stats = FeatureStats.fit(df, MOMENTUM_FEATURES)
    return ScalpingEnv(df, CostModel(), ActionMapper(), stats, **kwargs)


# ---------- actions ----------

def test_action_mapper_directions():
    m = ActionMapper()
    assert m.direction(FLAT) == 0
    assert m.direction(LONG_TIGHT) == 1
    assert m.direction(SHORT_TIGHT) == -1


def test_exit_levels_scale_with_volatility_and_are_clamped():
    # Zero-cost model isolates this to the vol_stop_scale/min/max clamp
    # math; the cost-floor behavior itself is covered separately below.
    zero_cost = CostModel(taker_fee=0.0, slippage_base=0.0, slippage_high_vol=0.0)
    m = ActionMapper(
        vol_stop_scale=3.0, min_stop_pct=0.002, max_stop_pct=0.02, reward_risk=1.5,
        cost_model=zero_cost,
    )
    tight = m.exit_levels(LONG_TIGHT, volatility=0.002, volatility_regime="low")   # 3 * 0.002 = 0.006
    assert tight.stop_loss_pct == pytest.approx(0.006)
    assert tight.take_profit_pct == pytest.approx(0.009)
    wide = m.exit_levels(LONG_WIDE, volatility=0.002, volatility_regime="low")     # 2.5x tight, clamped to max
    assert wide.stop_loss_pct == pytest.approx(0.015)
    assert m.exit_levels(LONG_TIGHT, volatility=1.0, volatility_regime="low").stop_loss_pct == 0.02   # clamp high
    assert m.exit_levels(LONG_TIGHT, volatility=1e-9, volatility_regime="low").stop_loss_pct == 0.002  # clamp low
    assert m.exit_levels(LONG_TIGHT, volatility=float("nan"), volatility_regime="low").stop_loss_pct >= 0.002


def test_exit_levels_stop_floor_clears_round_trip_cost():
    # A stop this tight would let a trade that closes exactly at
    # take-profit still net negative after fees + slippage — the
    # cost-derived floor must prevent that in every regime.
    cost = CostModel(taker_fee=0.001, slippage_base=0.001, slippage_high_vol=0.005)
    m = ActionMapper(
        vol_stop_scale=3.0, min_stop_pct=0.0005, max_stop_pct=0.02, reward_risk=1.5,
        cost_model=cost, cost_margin=1.5,
    )
    for regime in ("low", "medium", "high"):
        levels = m.exit_levels(LONG_TIGHT, volatility=1e-9, volatility_regime=regime)
        assert levels.take_profit_pct >= cost.round_trip_cost(regime) * 1.5 - 1e-12

    # Missing regime falls back to the worst-case (high-vol) cost, not the
    # cheapest — it must not silently under-price the stop.
    default_levels = m.exit_levels(LONG_TIGHT, volatility=1e-9)
    high_levels = m.exit_levels(LONG_TIGHT, volatility=1e-9, volatility_regime="high")
    assert default_levels.stop_loss_pct == pytest.approx(high_levels.stop_loss_pct)


# ---------- feature stats ----------

def test_feature_stats_roundtrip_and_normalization(featured_df):
    stats = FeatureStats.fit(featured_df, MOMENTUM_FEATURES)
    restored = FeatureStats.from_dict(stats.to_dict())
    values = featured_df[MOMENTUM_FEATURES].to_numpy(dtype=float)
    np.testing.assert_allclose(stats.transform(values), restored.transform(values))
    normalized = stats.transform(values)
    finite = normalized[np.isfinite(normalized).all(axis=1)]
    assert np.abs(np.nanmean(finite, axis=0)).max() < 0.5  # roughly centered


# ---------- environment ----------

def test_env_reset_skips_indicator_warmup(featured_df):
    env = make_env(featured_df)
    obs = env.reset()
    assert env.i >= env._first_valid
    assert obs.shape == (env.observation_dim,)
    assert np.isfinite(obs).all()


def test_env_entry_executes_at_next_bar_open_with_slippage(featured_df):
    env = make_env(featured_df)
    env.reset()
    i_before = env.i
    env.step(LONG_TIGHT)
    assert env.position is not None
    assert env.position.entry_index == i_before + 1
    expected = CostModel().entry_price(env._open[i_before + 1], 1, env._regime[i_before + 1])
    assert env.position.entry_price == pytest.approx(expected)


def test_env_position_sizing_respects_one_percent_risk(featured_df):
    env = make_env(featured_df)
    env.reset()
    env.step(LONG_TIGHT)
    pos = env.position
    risk = pos.size * pos.stop_distance
    assert risk <= 10_000.0 * 0.01 * 1.0001  # never above 1% of equity


def test_env_rejects_risk_above_cap(featured_df):
    stats = FeatureStats.fit(featured_df, MOMENTUM_FEATURES)
    with pytest.raises(ValueError):
        ScalpingEnv(featured_df, CostModel(), ActionMapper(), stats, risk_per_trade=0.05)


def test_env_flat_action_closes_open_position(featured_df):
    env = make_env(featured_df)
    env.reset()
    env.step(LONG_TIGHT)
    assert env.position is not None
    steps = 0
    while env.position is not None and steps < 10:
        _, reward, _, info = env.step(FLAT)
        steps += 1
    assert env.position is None
    assert len(env.closed_trades) == 1


def test_env_reward_on_close_matches_trade_r_multiple(featured_df):
    env = make_env(featured_df)
    env.reset()
    env.step(LONG_TIGHT)
    close_reward, closed_trade, done = None, None, False
    while closed_trade is None and not done:
        _, reward, done, info = env.step(LONG_TIGHT)  # hold same direction
        if "closed_trade" in info:
            close_reward, closed_trade = reward, info["closed_trade"]
    assert closed_trade is not None, "position never closed"
    # Reward on the closing step = clipped R-multiple (after fees+slippage).
    assert close_reward == pytest.approx(np.clip(closed_trade["r_multiple"], -5.0, 5.0))


def test_env_equity_reflects_closed_trade_pnl(featured_df):
    env = make_env(featured_df)
    env.reset()
    env.step(LONG_TIGHT)
    done = False
    while env.position is not None and not done:
        _, _, done, _ = env.step(LONG_TIGHT)
    assert env.equity == pytest.approx(10_000.0 + sum(t["pnl"] for t in env.closed_trades))


def test_env_episode_terminates_at_end(featured_df):
    env = make_env(featured_df)
    env.reset(start=len(featured_df) - 50)
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(FLAT)
        steps += 1
    assert steps <= 50


# ---------- network / replay / agent ----------

def test_dueling_network_output_shape():
    net = DuelingQNetwork(observation_dim=10, n_actions=N_ACTIONS)
    q = net(torch.zeros(7, 10))
    assert q.shape == (7, N_ACTIONS)
    assert torch.isfinite(q).all()


def test_replay_buffer_ring_semantics():
    buf = ReplayBuffer(capacity=5, observation_dim=3)
    for k in range(8):
        buf.push(np.full(3, k, dtype=np.float32), k % 2, float(k), np.zeros(3, dtype=np.float32), False)
    assert len(buf) == 5
    obs, actions, rewards, next_obs, dones = buf.sample(4)
    assert obs.shape == (4, 3)
    assert rewards.min() >= 3.0  # oldest entries were overwritten


def test_agent_act_returns_valid_actions_and_greedy_is_deterministic():
    agent = DQNAgent(observation_dim=8, n_actions=N_ACTIONS, config=DQNConfig(seed=1))
    obs = np.random.default_rng(0).normal(size=8).astype(np.float32)
    actions = {agent.act(obs, greedy=True) for _ in range(10)}
    assert len(actions) == 1  # deterministic under greedy
    assert all(0 <= agent.act(obs) < N_ACTIONS for _ in range(50))


def test_agent_learning_step_changes_parameters():
    agent = DQNAgent(observation_dim=6, n_actions=N_ACTIONS, config=DQNConfig(warmup_steps=10, batch_size=8, seed=2))
    rng = np.random.default_rng(0)
    before = [p.detach().clone() for p in agent.online.parameters()]
    loss = None
    for _ in range(60):
        loss = agent.observe(
            rng.normal(size=6).astype(np.float32), int(rng.integers(N_ACTIONS)),
            float(rng.normal()), rng.normal(size=6).astype(np.float32), False,
        )
    assert loss is not None and np.isfinite(loss)
    after = list(agent.online.parameters())
    assert any(not torch.equal(b, a.detach()) for b, a in zip(before, after))


def test_agent_state_dict_roundtrip():
    agent = DQNAgent(observation_dim=6, n_actions=N_ACTIONS, config=DQNConfig(seed=3))
    clone = DQNAgent(observation_dim=6, n_actions=N_ACTIONS, config=DQNConfig(seed=99))
    clone.load_state_dict(agent.state_dict())
    obs = np.ones(6, dtype=np.float32)
    assert agent.act(obs, greedy=True) == clone.act(obs, greedy=True)


# ---------- backtester adapter ----------

def test_dqn_strategy_runs_through_backtester(featured_df):
    stats = FeatureStats.fit(featured_df, MOMENTUM_FEATURES)
    agent = DQNAgent(observation_dim=len(MOMENTUM_FEATURES) + 3, n_actions=N_ACTIONS, config=DQNConfig(seed=4))
    strategy = DQNStrategy(agent, stats, ActionMapper())
    engine = EventDrivenBacktester(CostModel(), initial_equity=10_000)
    result = engine.run(featured_df, strategy)
    # Untrained agent may or may not trade; whatever it does must be well-formed.
    assert result.final_equity > 0
    for t in result.trades:
        assert t.direction in (1, -1)
        assert t.fees > 0
