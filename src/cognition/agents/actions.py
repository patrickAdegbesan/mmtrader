"""Discrete action space shared by the RL agents.

Stop-loss and take-profit are OUTPUTS of the model, per the spec: each
entry action carries a tightness multiplier applied to a
volatility-adjusted base stop. The Risk Engine (Milestone 5) clamps
whatever comes out of here; the env/backtester already enforce the 1%
equity-risk sizing cap regardless of the stop width chosen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    [min_stop_pct, max_stop_pct]. Take-profit = reward_risk x stop.
    """

    vol_stop_scale: float = 3.0
    min_stop_pct: float = 0.002
    max_stop_pct: float = 0.02
    reward_risk: float = 1.5

    def direction(self, action: int) -> int:
        return _ACTION_DIRECTION[action]

    def exit_levels(self, action: int, volatility: float) -> ExitLevels:
        if action == FLAT:
            raise ValueError("FLAT action has no exit levels")
        vol = volatility if np.isfinite(volatility) and volatility > 0 else self.min_stop_pct
        base_stop = float(np.clip(vol * self.vol_stop_scale, self.min_stop_pct, self.max_stop_pct))
        stop = base_stop * _ACTION_STOP_MULT[action]
        stop = float(np.clip(stop, self.min_stop_pct, self.max_stop_pct))
        return ExitLevels(stop_loss_pct=stop, take_profit_pct=stop * self.reward_risk)
