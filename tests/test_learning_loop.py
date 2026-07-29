"""Milestone-3 acceptance test: the learning loop measurably improves
the agent.

Setup: a strongly-trending synthetic market (persistent upward drift,
modest noise) where going long is reliably profitable after costs and
going short/overtrading reliably is not. A learning agent must discover
this; an agent that doesn't learn won't beat its own untrained baseline.

Seeds are fixed end-to-end for reproducibility.
"""
import numpy as np
import pandas as pd
import pytest

from cognition.agents.actions import ActionMapper
from cognition.agents.dqn import DQNConfig
from cognition.agents.trainer import AgentTrainer, TrainerConfig
from cognition.backtest.costs import CostModel
from cognition.features.engineering import extract_features
from cognition.learning.registry import ModelRegistry
from cognition.utils.timeframes import timeframe_to_ms


def make_trending_market(n=5000, drift=0.0008, noise=0.0004, seed=0):
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0, noise, n)
    close = 100 * np.cumprod(1 + returns)
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0, noise, n)) * close
    df = pd.DataFrame({
        "timestamp": 1_700_000_000_000 + np.arange(n, dtype=np.int64) * timeframe_to_ms("1m"),
        "open": open_,
        "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick,
        "close": close,
        "volume": rng.uniform(1, 10, n),
    })
    return extract_features(df)


@pytest.fixture(scope="module")
def training_run(tmp_path_factory):
    df = make_trending_market()
    registry = ModelRegistry(tmp_path_factory.mktemp("models"), "momentum_test")
    config = TrainerConfig(
        episodes=24,
        episode_bars=400,
        eval_every=8,
        eval_fraction=0.2,
        seed=7,
        dqn=DQNConfig(
            hidden=32, buffer_capacity=10_000, batch_size=32, warmup_steps=200,
            target_sync_every=250, epsilon_decay_steps=3_000, seed=7,
        ),
    )
    trainer = AgentTrainer(df, CostModel(), ActionMapper(), registry, config)
    history = trainer.train()
    return trainer, registry, history


def test_learning_improves_eval_reward(training_run):
    _, _, history = training_run
    baseline, final = history[0], history[-1]
    assert final.total_reward > baseline.total_reward, (
        f"agent did not improve: baseline {baseline.total_reward:.2f} -> final {final.total_reward:.2f}"
    )


def test_trained_agent_is_profitable_on_easy_trending_market(training_run):
    """Profitability = equity after all costs. (total_reward also contains
    holding-penalty shaping, so it is NOT the profit measure.)"""
    _, _, history = training_run
    final = history[-1]
    assert final.num_trades > 0
    assert final.final_equity > 10_000.0


def test_checkpoints_are_versioned_in_registry(training_run):
    _, registry, history = training_run
    n_evals = len(history) - 1  # baseline isn't checkpointed
    assert len(registry.list_versions()) == n_evals
    # LATEST tracks the best-by-terminal-equity checkpoint seen during
    # training, not simply the chronologically last one — DQN training is
    # non-monotonic, so the last episode is not guaranteed to be the best.
    evals = history[1:]
    best_idx = max(range(len(evals)), key=lambda i: evals[i].final_equity)
    assert registry.latest_version() == registry.list_versions()[best_idx]
    _, meta = registry.load()
    assert "eval" in meta and "feature_stats" in meta


def test_train_eval_split_has_no_overlap(training_run):
    trainer, _, _ = training_run
    assert trainer.train_df["timestamp"].max() < trainer.eval_df["timestamp"].min()
