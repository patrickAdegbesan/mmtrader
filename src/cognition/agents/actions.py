"""Discrete action space shared by the RL agents.

Stop-loss and take-profit are OUTPUTS of the model, per the spec: each
entry action carries a tightness multiplier applied to a
volatility-adjusted base stop. The Risk Engine (Milestone 5) clamps
whatever comes out of here; the env/backtester already enforce the 1%
equity-risk sizing cap regardless of the stop width chosen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cognition.backtest.costs import HIGH_VOL_REGIME, CostModel

FLAT, LONG_TIGHT, LONG_WIDE, SHORT_TIGHT, SHORT_WIDE = range(5)
N_ACTIONS = 5

_ACTION_DIRECTION = {FLAT: 0, LONG_TIGHT: 1, LONG_WIDE: 1, SHORT_TIGHT: -1, SHORT_WIDE: -1}
_ACTION_STOP_MULT = {LONG_TIGHT: 1.0, LONG_WIDE: 2.5, SHORT_TIGHT: 1.0, SHORT_WIDE: 2.5}


@dataclass(frozen=True)
class ExitLevels:
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(frozen=True)
class ActionMapper:
    """Maps (action id, current volatility) -> direction + SL/TP levels.

    Base stop = vol_stop_scale x rolling volatility, clamped to
    [effective_min_stop_pct, max_stop_pct]. Take-profit = reward_risk x stop.

    effective_min_stop_pct is NOT just min_stop_pct: a stop that's too
    tight relative to round-trip costs means a trade that hits take-profit
    dead-on still loses money after fees + slippage. At the old fixed
    0.2% floor, real BTC 1m volatility clamped ~68% of TIGHT stops to that
    floor, and every one of those "wins" netted negative in low/medium
    vol (100% of the time) and nearly all of them in high vol (98.5%) —
    see backtest_ensemble diagnostic, 2026-07-29. The floor is now
    max(min_stop_pct, cost_margin x round_trip_cost / reward_risk), which
    guarantees a trade that closes exactly at take-profit nets a real
    profit (by cost_margin) after costs, in whatever regime it trades.
    """

    vol_stop_scale: float = 3.0
    min_stop_pct: float = 0.002
    max_stop_pct: float = 0.02
    reward_risk: float = 1.5
    cost_model: CostModel = field(default_factory=CostModel)
    cost_margin: float = 1.5

    def direction(self, action: int) -> int:
        return _ACTION_DIRECTION[action]

    def _effective_min_stop_pct(self, volatility_regime: object) -> float:
        cost_floor = self.cost_model.round_trip_cost(volatility_regime) * self.cost_margin / self.reward_risk
        return max(self.min_stop_pct, cost_floor)

    def exit_levels(self, action: int, volatility: float, volatility_regime: object = None) -> ExitLevels:
        if action == FLAT:
            raise ValueError("FLAT action has no exit levels")
        # Unknown regime falls back to the worst-case (high-vol) cost floor
        # rather than the cheapest one, so a missing regime can't silently
        # under-price the stop.
        regime = volatility_regime if volatility_regime is not None else HIGH_VOL_REGIME
        min_stop_pct = self._effective_min_stop_pct(regime)
        vol = volatility if np.isfinite(volatility) and volatility > 0 else min_stop_pct
        base_stop = float(np.clip(vol * self.vol_stop_scale, min_stop_pct, self.max_stop_pct))
        stop = base_stop * _ACTION_STOP_MULT[action]
        stop = float(np.clip(stop, min_stop_pct, self.max_stop_pct))
        return ExitLevels(stop_loss_pct=stop, take_profit_pct=stop * self.reward_risk)
