"""Retraining promotion gate: no silent updates, auto-rollback semantics."""
import numpy as np
import pandas as pd
import pytest

from cognition.agents.actions import ActionMapper
from cognition.agents.dqn import DQNConfig
from cognition.agents.specs import AGENT_SPECS
from cognition.agents.trainer import TrainerConfig
from cognition.backtest.costs import CostModel
from cognition.features.engineering import extract_features
from cognition.learning.registry import ModelRegistry
from cognition.learning.retrain import retrain_agent, should_promote
from cognition.utils.timeframes import timeframe_to_ms


# ---------- the gate itself ----------

def test_first_version_always_promotes():
    promoted, reason = should_promote(None, new_score=123.0)
    assert promoted and "first version" in reason


def test_better_or_equal_promotes():
    assert should_promote(10_000.0, 10_100.0)[0]
    assert should_promote(10_000.0, 10_000.0)[0]


def test_worse_is_refused():
    promoted, reason = should_promote(10_000.0, 9_500.0)
    assert not promoted
    assert "UNDERPERFORMS" in reason


def test_tolerance_allows_small_regressions_only():
    assert should_promote(10_000.0, 9_990.0, tolerance=50.0)[0]
    assert not should_promote(10_000.0, 9_900.0, tolerance=50.0)[0]


# ---------- end-to-end retrain flow ----------

def make_trending_df(n=4000, drift=0.0008, seed=0):
    rng = np.random.default_rng(seed)
    returns = drift + rng.normal(0, 0.0004, n)
    close = 100 * np.cumprod(1 + returns)
    open_ = np.concatenate([[100.0], close[:-1]])
    wick = np.abs(rng.normal(0, 0.0004, n)) * close
    df = pd.DataFrame({
        "timestamp": 1_700_006_400_000 + np.arange(n, dtype=np.int64) * timeframe_to_ms("1m"),
        "open": open_, "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick, "close": close,
        "volume": rng.uniform(1, 10, n),
    })
    return extract_features(df)


def small_trainer_config(episodes=8, seed=7):
    return TrainerConfig(
        episodes=episodes, episode_bars=300, eval_every=episodes,
        eval_fraction=0.2, seed=seed,
        dqn=DQNConfig(hidden=32, buffer_capacity=5_000, batch_size=32,
                      warmup_steps=100, target_sync_every=200,
                      epsilon_decay_steps=1_500, seed=seed),
    )


@pytest.fixture(scope="module")
def retrain_setup(tmp_path_factory):
    df = make_trending_df()
    spec = AGENT_SPECS["momentum"]
    registry = ModelRegistry(tmp_path_factory.mktemp("models"), "momentum")
    mapper = ActionMapper()
    cost = CostModel()

    first = retrain_agent(df, spec, registry, cost, mapper, small_trainer_config(seed=7))
    second = retrain_agent(df, spec, registry, cost, mapper, small_trainer_config(seed=8))
    return registry, first, second


def test_first_retrain_creates_and_promotes(retrain_setup):
    registry, first, _ = retrain_setup
    assert first.promoted
    assert first.old_score is None
    assert first.new_version in registry.list_versions()


def test_second_retrain_is_gated_against_production(retrain_setup):
    registry, first, second = retrain_setup
    # Whatever the outcome, the gate compared against production...
    assert second.old_score is not None
    # ...saved the new version immutably...
    assert second.new_version in registry.list_versions()
    assert second.new_version != first.new_version
    # ...and the ACTIVE version reflects the decision exactly.
    if second.promoted:
        assert registry.latest_version() == second.new_version
    else:
        assert registry.latest_version() == first.active_version
        assert second.new_score < second.old_score  # refused for cause


def test_refused_promotion_leaves_production_active(tmp_path):
    """Force a refusal deterministically: a huge negative tolerance means
    any new score is 'worse'. Production must stay active."""
    df = make_trending_df(n=3000, seed=3)
    spec = AGENT_SPECS["momentum"]
    registry = ModelRegistry(tmp_path, "momentum")
    mapper, cost = ActionMapper(), CostModel()

    first = retrain_agent(df, spec, registry, cost, mapper, small_trainer_config(episodes=4, seed=1))
    assert first.promoted
    production = registry.latest_version()

    second = retrain_agent(
        df, spec, registry, cost, mapper, small_trainer_config(episodes=4, seed=2),
        tolerance=-1e12,  # nothing can pass this gate
    )
    assert not second.promoted
    assert registry.latest_version() == production          # auto-rollback
    assert second.new_version in registry.list_versions()   # but kept on disk
