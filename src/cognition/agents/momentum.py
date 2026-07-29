"""Momentum agent: DQN policy over momentum features, plus the Strategy
adapter that lets a trained agent run inside the Milestone-2 backtester
(walk-forward, Monte Carlo, regime reports — all of it works on agents
exactly as it did on the dummy strategy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cognition.agents.actions import ActionMapper, FLAT
from cognition.agents.dqn import DQNAgent, DQNConfig
from cognition.agents.env import MOMENTUM_FEATURES, FeatureStats
from cognition.backtest.simulator import Signal
from cognition.learning.registry import ModelRegistry


def load_agent(registry: ModelRegistry, version: str | None = None) -> tuple[DQNAgent, FeatureStats, dict]:
    """Load any DQN agent checkpoint (momentum, mean-reversion, ...) from
    its registry along with the feature-normalization stats it was
    trained with."""
    state, meta = registry.load(version)
    config = DQNConfig(**state["config"])
    agent = DQNAgent(state["observation_dim"], state["n_actions"], config)
    agent.load_state_dict(state)
    stats = FeatureStats.from_dict(meta["feature_stats"])
    return agent, stats, meta


load_momentum_agent = load_agent  # historical name from Milestone 3


class DQNStrategy:
    """Adapts a trained DQNAgent to the backtester's Strategy protocol.

    The engine owns execution (latency, slippage, fees, sizing); the agent
    only proposes direction + SL/TP, mirroring how the live system will
    separate decision from execution.
    """

    def __init__(
        self,
        agent: DQNAgent,
        stats: FeatureStats,
        action_mapper: ActionMapper,
        feature_columns: list[str] = MOMENTUM_FEATURES,
        max_holding_bars: int = 120,
    ):
        self.agent = agent
        self.stats = stats
        self.mapper = action_mapper
        self.feature_columns = feature_columns
        self.max_holding_bars = max_holding_bars
        self._features: np.ndarray | None = None
        self._vol: np.ndarray | None = None
        self._vol_regime: np.ndarray | None = None
        self._valid: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # training happens in AgentTrainer, not per-window here

    def prepare(self, df: pd.DataFrame) -> None:
        self._features = self.stats.transform(df[self.feature_columns].to_numpy(dtype=float))
        self._vol = df["volatility"].to_numpy(dtype=float)
        self._vol_regime = (
            df["volatility_regime"].to_numpy(dtype=object)
            if "volatility_regime" in df.columns else np.full(len(df), None, dtype=object)
        )
        self._valid = np.isfinite(self._features).all(axis=1)

    def signal(self, i: int) -> Signal | None:
        if not self._valid[i]:
            return None
        # The engine tracks position state; for signal generation we present
        # the agent a flat-position view and let the engine handle flips.
        obs = np.concatenate([self._features[i], [0.0, 0.0, 0.0]]).astype(np.float32)
        action = self.agent.act(obs, greedy=True)
        if action == FLAT:
            return None
        direction = self.mapper.direction(action)
        levels = self.mapper.exit_levels(action, self._vol[i], self._vol_regime[i])
        return Signal(
            direction=direction,
            stop_loss_pct=levels.stop_loss_pct,
            take_profit_pct=levels.take_profit_pct,
            max_holding_bars=self.max_holding_bars,
        )
