"""Training loop for a single agent, with the learning-loop verification
the milestone demands: periodic greedy evaluation on a held-out slice,
metrics history, and versioned checkpoints via the ModelRegistry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cognition.agents.actions import N_ACTIONS, ActionMapper
from cognition.agents.dqn import DQNAgent, DQNConfig
from cognition.agents.env import MOMENTUM_FEATURES, FeatureStats, ScalpingEnv
from cognition.backtest.costs import CostModel
from cognition.learning.registry import ModelRegistry
from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("trainer")


@dataclass
class TrainerConfig:
    episodes: int = 60
    episode_bars: int = 720          # ~12h of 1m bars per episode
    eval_every: int = 10             # episodes between greedy evals
    eval_fraction: float = 0.2       # tail fraction of data held out for eval
    seed: int = 0
    # When False, checkpoints are saved but LATEST is not moved — used by
    # scheduled retraining, where promotion is a separate gated decision.
    activate_checkpoints: bool = True
    dqn: DQNConfig = field(default_factory=DQNConfig)


@dataclass
class EvalResult:
    episode: int
    total_reward: float
    num_trades: int
    avg_r: float
    win_rate: float
    final_equity: float


def evaluate(env: ScalpingEnv, agent: DQNAgent) -> EvalResult:
    """Greedy rollout over the full eval slice."""
    obs = env.reset()
    total_reward, done = 0.0, False
    while not done:
        obs, reward, done, _ = env.step(agent.act(obs, greedy=True))
        total_reward += reward
    trades = env.closed_trades
    rs = [t["r_multiple"] for t in trades]
    return EvalResult(
        episode=-1,
        total_reward=total_reward,
        num_trades=len(trades),
        avg_r=float(np.mean(rs)) if rs else 0.0,
        win_rate=float(np.mean([t["pnl"] > 0 for t in trades])) if trades else 0.0,
        final_equity=env.equity,
    )


class AgentTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        cost_model: CostModel,
        action_mapper: ActionMapper,
        registry: ModelRegistry,
        config: TrainerConfig,
        feature_columns: list[str] = MOMENTUM_FEATURES,
    ):
        self.config = config
        self.registry = registry
        self._rng = np.random.default_rng(config.seed)

        split = int(len(df) * (1 - config.eval_fraction))
        self.train_df = df.iloc[:split].reset_index(drop=True)
        self.eval_df = df.iloc[split:].reset_index(drop=True)

        # Normalization stats come from the TRAIN slice only (no leakage),
        # and are persisted with every checkpoint so inference matches.
        self.stats = FeatureStats.fit(self.train_df, feature_columns)
        self.train_env = ScalpingEnv(self.train_df, cost_model, action_mapper, self.stats, feature_columns)
        self.eval_env = ScalpingEnv(self.eval_df, cost_model, action_mapper, self.stats, feature_columns)

        self.agent = DQNAgent(self.train_env.observation_dim, N_ACTIONS, config.dqn)
        self.history: list[EvalResult] = []

    def _random_episode_window(self) -> tuple[int, int]:
        n = len(self.train_df)
        ep_bars = min(self.config.episode_bars, n - self.train_env._first_valid - 2)
        start_max = n - ep_bars - 1
        start = int(self._rng.integers(self.train_env._first_valid, max(start_max, self.train_env._first_valid + 1)))
        return start, start + ep_bars

    def train(self) -> list[EvalResult]:
        baseline = evaluate(self.eval_env, self.agent)
        baseline.episode = 0
        self.history.append(baseline)
        log_with_fields(logger, 20, "Baseline eval (untrained)", **baseline.__dict__)

        for episode in range(1, self.config.episodes + 1):
            start, end = self._random_episode_window()
            obs = self.train_env.reset(start=start, end=end)
            done = False
            while not done:
                action = self.agent.act(obs)
                next_obs, reward, done, _ = self.train_env.step(action)
                self.agent.observe(obs, action, reward, next_obs, done)
                obs = next_obs

            if episode % self.config.eval_every == 0 or episode == self.config.episodes:
                result = evaluate(self.eval_env, self.agent)
                result.episode = episode
                self.history.append(result)
                version = self.registry.save(
                    self.agent.state_dict(),
                    metadata={
                        "episode": episode,
                        "eval": result.__dict__,
                        "feature_stats": self.stats.to_dict(),
                        "trainer_config": {
                            "episodes": self.config.episodes,
                            "episode_bars": self.config.episode_bars,
                            "seed": self.config.seed,
                        },
                    },
                    activate=self.config.activate_checkpoints,
                )
                log_with_fields(logger, 20, "Eval checkpoint", version=version, **result.__dict__)

        return self.history
