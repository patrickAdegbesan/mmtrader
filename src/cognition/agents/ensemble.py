"""The 5-agent ensemble as a backtester Strategy.

Per bar: each of the four market agents votes (direction + confidence
from its Q-values + its preferred volatility-adjusted SL/TP). The
meta-learner supplies per-regime weights. The combined score is

    score = sum_i weight_i x confidence_i x direction_i

and a trade is proposed when |score| clears the vote threshold. SL/TP is
the confidence-weighted average of the levels proposed by the agents
voting in the winning direction.

After every closed trade the engine calls on_trade_closed(), which
updates the meta-learner online (regime memory) and the session tracker.
Every decision is appended to decision_log — the audit trail Guardian B
requires ("why did the ensemble vote the way it did").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from cognition.agents.actions import FLAT, ActionMapper
from cognition.agents.dqn import DQNAgent
from cognition.agents.env import FeatureStats
from cognition.agents.meta import AgentVote, MetaLearner, SessionPerformanceTracker
from cognition.backtest.simulator import Signal, Trade


@dataclass
class EnsembleMember:
    name: str
    agent: DQNAgent
    stats: FeatureStats
    feature_columns: list[str]


class EnsembleStrategy:
    def __init__(
        self,
        members: list[EnsembleMember],
        action_mapper: ActionMapper,
        meta: MetaLearner | None = None,
        vote_threshold: float = 0.15,
        max_holding_bars: int = 120,
        keep_decision_log: bool = True,
        online_learning: bool = False,
    ):
        self.members = members
        self.mapper = action_mapper
        self.meta = meta or MetaLearner([m.name for m in members])
        self.vote_threshold = vote_threshold
        self.max_holding_bars = max_holding_bars
        self.keep_decision_log = keep_decision_log
        # When True (paper/live trading), each closed trade also feeds a
        # trade-level transition back into every voting agent's replay
        # buffer: (obs at entry, action it chose, post-cost R) — the
        # per-trade learning loop from the spec, on top of the
        # meta-learner reweighting.
        self.online_learning = online_learning

        self.session_tracker = SessionPerformanceTracker()
        self.decision_log: list[dict] = []
        self._pending_votes: dict[str, AgentVote] | None = None
        self._pending_regime: object = None
        self._pending_obs: dict[str, np.ndarray] | None = None
        self._features: dict[str, np.ndarray] = {}
        self._valid: np.ndarray | None = None
        self._vol: np.ndarray | None = None
        self._vol_regime: np.ndarray | None = None
        self._regime: np.ndarray | None = None
        self._session: np.ndarray | None = None

    # ---- Strategy protocol ------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> None:
        pass  # agents are trained by AgentTrainer; nothing to fit per window

    def prepare(self, df: pd.DataFrame) -> None:
        n = len(df)
        valid = np.ones(n, dtype=bool)
        for m in self.members:
            feats = m.stats.transform(df[m.feature_columns].to_numpy(dtype=float))
            self._features[m.name] = feats
            valid &= np.isfinite(feats).all(axis=1)
        self._valid = valid
        self._vol = df["volatility"].to_numpy(dtype=float)
        self._vol_regime = (
            df["volatility_regime"].to_numpy(dtype=object)
            if "volatility_regime" in df.columns else np.full(n, None, dtype=object)
        )
        self._regime = (
            df["market_regime"].to_numpy(dtype=object)
            if "market_regime" in df.columns else np.full(n, "unknown", dtype=object)
        )
        self._session = (
            df["session"].to_numpy(dtype=object)
            if "session" in df.columns else np.full(n, None, dtype=object)
        )

    def _member_vote(self, member: EnsembleMember, i: int) -> tuple[AgentVote, np.ndarray]:
        obs = np.concatenate([self._features[member.name][i], [0.0, 0.0, 0.0]]).astype(np.float32)
        with torch.no_grad():
            q = member.agent.online(torch.from_numpy(obs).unsqueeze(0)).squeeze(0)
        action = int(q.argmax().item())
        # Confidence = softmax probability mass of the chosen action.
        probs = torch.softmax(q, dim=-1)
        confidence = float(probs[action].item())
        direction = self.mapper.direction(action)
        if direction == 0:
            return AgentVote(direction=0, confidence=confidence, action=action), obs
        levels = self.mapper.exit_levels(action, self._vol[i], self._vol_regime[i])
        return AgentVote(
            direction=direction, confidence=confidence,
            stop_loss_pct=levels.stop_loss_pct, take_profit_pct=levels.take_profit_pct,
            action=action,
        ), obs

    def signal(self, i: int) -> Signal | None:
        if not self._valid[i]:
            return None
        regime = self._regime[i]
        weights = self.meta.weights(regime)
        votes: dict[str, AgentVote] = {}
        observations: dict[str, np.ndarray] = {}
        for m in self.members:
            votes[m.name], observations[m.name] = self._member_vote(m, i)

        score = sum(weights[name] * v.confidence * v.direction for name, v in votes.items())
        direction = int(np.sign(score)) if abs(score) >= self.vote_threshold else 0

        if self.keep_decision_log:
            self.decision_log.append({
                "index": i, "regime": regime, "session": self._session[i],
                "score": float(score), "direction": direction,
                "weights": dict(weights),
                "votes": {n: (v.direction, round(v.confidence, 4)) for n, v in votes.items()},
            })

        if direction == 0:
            return None

        # SL/TP: confidence-weighted average over agents voting this way.
        agreeing = [(weights[n] * v.confidence, v) for n, v in votes.items() if v.direction == direction]
        total = sum(w for w, _ in agreeing)
        stop = sum(w * v.stop_loss_pct for w, v in agreeing) / total
        tp = sum(w * v.take_profit_pct for w, v in agreeing) / total

        self._pending_votes = votes
        self._pending_regime = regime
        self._pending_obs = observations
        return Signal(
            direction=direction, stop_loss_pct=stop, take_profit_pct=tp,
            max_holding_bars=self.max_holding_bars,
        )

    # ---- learning-loop hook (called by the engine on every close) ----

    def on_trade_closed(self, trade: Trade) -> None:
        if self._pending_votes is not None:
            self.meta.update(
                votes=self._pending_votes,
                executed_direction=trade.direction,
                r_multiple=trade.r_multiple,
                regime=trade.market_regime if trade.market_regime is not None else self._pending_regime,
            )
            if self.online_learning and self._pending_obs is not None:
                reward = float(np.clip(trade.r_multiple, -5.0, 5.0))
                for m in self.members:
                    vote = self._pending_votes.get(m.name)
                    obs = self._pending_obs.get(m.name)
                    if vote is None or obs is None:
                        continue
                    # Trade-level transition: the observation the agent
                    # voted on, the action it chose, the post-cost R the
                    # position realized. Terminal (done) — trades are
                    # episodic events in live learning.
                    m.agent.observe(obs, vote.action, reward, obs, True)
            self._pending_votes = None
            self._pending_obs = None
        self.session_tracker.update(trade.market_regime, trade.session, trade.pnl, trade.r_multiple)

    def last_votes_snapshot(self) -> dict:
        """Serializable copy of the votes behind the most recent entry
        signal — journaled with the trade for auditability."""
        if self._pending_votes is None:
            return {}
        return {
            name: {"direction": v.direction, "confidence": round(v.confidence, 4)}
            for name, v in self._pending_votes.items()
        }
