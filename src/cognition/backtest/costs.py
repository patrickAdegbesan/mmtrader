"""Cost model — the piece that keeps backtests honest.

Models the four cost sources the architecture doc demands:
- taker fees (Bybit 0.075%–0.1%; we default to the conservative 0.1%)
- slippage, regime-aware: 0.1% baseline widened to 0.5% in high-volatility
  candles (keyed off the `volatility_regime` feature column)
- order latency: a signal generated at the close of bar i executes at the
  open of bar i+latency_bars, never inside the same bar (no lookahead)
- partial fills: `fill_ratio` scales down the size actually obtained

Slippage is always applied in the adverse direction: entries fill worse
than the ideal price, exits fill worse than the ideal price.
"""
from __future__ import annotations

from dataclasses import dataclass

HIGH_VOL_REGIME = "high"


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.001          # 0.1% — conservative end of Bybit's range
    slippage_base: float = 0.001      # 0.1%
    slippage_high_vol: float = 0.005  # 0.5%
    latency_bars: int = 1
    fill_ratio: float = 1.0           # fraction of intended size actually filled

    def __post_init__(self) -> None:
        if not 0 < self.fill_ratio <= 1:
            raise ValueError("fill_ratio must be in (0, 1]")
        if self.latency_bars < 1:
            raise ValueError("latency_bars must be >= 1 (same-bar fills are lookahead)")

    def slippage_rate(self, volatility_regime: object) -> float:
        if volatility_regime == HIGH_VOL_REGIME:
            return self.slippage_high_vol
        return self.slippage_base

    def entry_price(self, ideal_price: float, direction: int, volatility_regime: object) -> float:
        """Adverse fill on entry: longs buy higher, shorts sell lower."""
        slip = self.slippage_rate(volatility_regime)
        return ideal_price * (1 + direction * slip)

    def exit_price(self, ideal_price: float, direction: int, volatility_regime: object) -> float:
        """Adverse fill on exit: longs sell lower, shorts buy back higher."""
        slip = self.slippage_rate(volatility_regime)
        return ideal_price * (1 - direction * slip)

    def fee(self, notional: float) -> float:
        return abs(notional) * self.taker_fee
