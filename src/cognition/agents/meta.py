"""Agent 5 — the Meta-Learner ("Head Trader").

Does not read the market directly. It watches the rolling performance of
Agents 1-4 and dynamically weights their votes, with REGIME MEMORY:
performance scores are kept per market regime, so momentum's vote counts
more in trends while mean-reversion dominates ranges — learned, not
hard-coded.

Online update after EVERY closed trade (spec requirement):
    contribution_i = alignment_i x confidence_i x trade_R
where alignment_i is +1 if agent i voted with the executed direction,
-1 if it voted against, 0 if it abstained. An agent that agreed with a
winner (or opposed a loser) gains score; agreeing with losers costs it.
Scores decay exponentially, so recent performance dominates.

Weights = softmax(score / temperature) per regime — always positive,
always sum to 1. A regime never seen yet gets uniform weights.

Deliberately transparent (EWMA + softmax rather than an opaque model):
every weight is auditable from the logged trade history, which Guardian
B's "why did the ensemble vote this way" requirement demands. A richer
contextual model can replace this class behind the same interface.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AgentVote:
    direction: int          # 1 long, -1 short, 0 abstain
    confidence: float       # [0, 1]
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    action: int = 0         # raw action id (for online RL updates)


@dataclass
class MetaLearner:
    agent_names: list[str]
    decay: float = 0.94
    temperature: float = 0.5

    def __post_init__(self):
        # regime -> agent -> EWMA score. Regime memory lives here.
        self._scores: dict[object, dict[str, float]] = defaultdict(
            lambda: {name: 0.0 for name in self.agent_names}
        )
        self.updates = 0

    def weights(self, regime: object) -> dict[str, float]:
        scores = self._scores[regime]
        values = np.array([scores[name] for name in self.agent_names])
        exp = np.exp((values - values.max()) / self.temperature)
        w = exp / exp.sum()
        return {name: float(w[k]) for k, name in enumerate(self.agent_names)}

    def update(
        self,
        votes: dict[str, AgentVote],
        executed_direction: int,
        r_multiple: float,
        regime: object,
    ) -> None:
        """Called after every closed trade."""
        r = float(np.clip(r_multiple, -5.0, 5.0))
        scores = self._scores[regime]
        for name in self.agent_names:
            vote = votes.get(name)
            alignment = 0.0
            if vote is not None and vote.direction != 0:
                alignment = float(np.sign(vote.direction * executed_direction)) * vote.confidence
            contribution = alignment * r
            scores[name] = self.decay * scores[name] + (1 - self.decay) * contribution
        self.updates += 1

    def snapshot(self) -> pd.DataFrame:
        """Weights per seen regime — for dashboards and audit logs."""
        rows = {regime: self.weights(regime) for regime in self._scores}
        return pd.DataFrame(rows).T


class SessionPerformanceTracker:
    """Regime x session performance memory (spec: 'a strategy that works
    in European hours but fails in Asian hours' must become visible).
    Tracks ensemble-level results per (market_regime, session) cell.
    """

    def __init__(self):
        self._cells: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "wins": 0, "total_r": 0.0, "total_pnl": 0.0})

    def update(self, regime: object, session: object, pnl: float, r_multiple: float) -> None:
        cell = self._cells[(regime, session)]
        cell["n"] += 1
        cell["wins"] += int(pnl > 0)
        cell["total_r"] += r_multiple
        cell["total_pnl"] += pnl

    def table(self) -> pd.DataFrame:
        if not self._cells:
            return pd.DataFrame()
        rows = []
        for (regime, session), cell in sorted(self._cells.items(), key=str):
            rows.append({
                "regime": regime, "session": session, "trades": cell["n"],
                "win_rate": cell["wins"] / cell["n"] if cell["n"] else 0.0,
                "avg_r": cell["total_r"] / cell["n"] if cell["n"] else 0.0,
                "total_pnl": cell["total_pnl"],
            })
        return pd.DataFrame(rows)
