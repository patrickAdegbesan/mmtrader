"""Scheduled full retraining with a promotion gate (Milestone 7).

Spec rule 8 in action: retraining NEVER silently replaces the production
model. The flow per agent is:

1. Load the current production version (registry LATEST), if any.
2. Warm-start a new training run from it (continuous learning) on the
   provided data — journal-accumulated history or fresh downloads.
3. Evaluate BOTH the production agent and the retrained agent greedily
   on the same held-out eval slice (post-cost equity).
4. Save the retrained agent as a new immutable version (never activated
   yet), with the comparison recorded in its metadata.
5. Promote (activate) the new version ONLY if it does not underperform
   the production version beyond the tolerance. Otherwise LATEST stays
   where it was — that is the auto-rollback: the bad version exists on
   disk for inspection but never serves.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cognition.agents.actions import ActionMapper
from cognition.agents.dqn import DQNAgent, DQNConfig
from cognition.agents.env import FeatureStats, ScalpingEnv
from cognition.agents.specs import AgentSpec
from cognition.agents.trainer import AgentTrainer, TrainerConfig, evaluate
from cognition.backtest.costs import CostModel
from cognition.learning.registry import ModelRegistry
from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("retrain")


@dataclass
class RetrainResult:
    agent_name: str
    new_version: str
    promoted: bool
    reason: str
    old_score: float | None      # production equity on eval slice (None = no production yet)
    new_score: float
    active_version: str


def should_promote(old_score: float | None, new_score: float, tolerance: float = 0.0) -> tuple[bool, str]:
    """Promote unless the retrained model underperforms production by
    more than `tolerance` (absolute equity units). First-ever training
    always promotes — there is nothing to protect yet."""
    if old_score is None:
        return True, "first version — no production model to compare against"
    if new_score >= old_score - tolerance:
        return True, f"new {new_score:.2f} >= production {old_score:.2f} - tolerance {tolerance:.2f}"
    return False, (
        f"UNDERPERFORMS: new {new_score:.2f} < production {old_score:.2f} - tolerance {tolerance:.2f}"
        " — production version stays active (auto-rollback)"
    )


def _eval_score(agent: DQNAgent, stats: FeatureStats, spec: AgentSpec,
                eval_df: pd.DataFrame, cost_model: CostModel, mapper: ActionMapper) -> float:
    env = ScalpingEnv(eval_df, cost_model, mapper, stats, feature_columns=spec.feature_columns)
    return float(evaluate(env, agent).final_equity)


def retrain_agent(
    df: pd.DataFrame,
    spec: AgentSpec,
    registry: ModelRegistry,
    cost_model: CostModel,
    mapper: ActionMapper,
    trainer_config: TrainerConfig,
    tolerance: float = 0.0,
) -> RetrainResult:
    trainer_config.activate_checkpoints = False  # promotion is gated below
    trainer = AgentTrainer(
        df, cost_model, mapper, registry, trainer_config,
        feature_columns=spec.feature_columns,
    )

    # Production baseline (and warm start) if a version exists.
    old_score: float | None = None
    production_version = registry.latest_version()
    if production_version is not None:
        state, meta = registry.load(production_version)
        old_stats = FeatureStats.from_dict(meta["feature_stats"])
        old_agent = DQNAgent(state["observation_dim"], state["n_actions"], DQNConfig(**state["config"]))
        old_agent.load_state_dict(state)
        old_score = _eval_score(old_agent, old_stats, spec, trainer.eval_df, cost_model, mapper)
        if state["observation_dim"] == trainer.agent.observation_dim:
            trainer.agent.load_state_dict(state)  # continuous learning

    trainer.train()
    new_version = registry.list_versions()[-1]
    new_score = _eval_score(trainer.agent, trainer.stats, spec, trainer.eval_df, cost_model, mapper)

    promoted, reason = should_promote(old_score, new_score, tolerance)
    if promoted:
        registry.activate(new_version)
    log_with_fields(
        logger, 20 if promoted else 30, "Retrain gate decision",
        agent=spec.name, new_version=new_version, promoted=promoted, reason=reason,
        old_score=old_score, new_score=round(new_score, 2),
    )
    return RetrainResult(
        agent_name=spec.name, new_version=new_version, promoted=promoted,
        reason=reason, old_score=old_score, new_score=new_score,
        active_version=registry.latest_version(),
    )
