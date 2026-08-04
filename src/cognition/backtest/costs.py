"""Cost model — the piece that keeps backtests honest.

Models the cost sources the architecture doc demands:
- fees, split by liquidity role. Taker crosses the spread and pays the
  higher rate; maker rests at the touch and pays the lower one (Bybit perp
  VIP0 is 0.055% taker / 0.02% maker, spot is 0.1% both ways). Defaults are
  the conservative taker end.
- slippage, regime-aware: 0.1% baseline widened to 0.5% in high-volatility
  candles (keyed off the `volatility_regime` feature column)
- order latency: a signal generated at the close of bar i executes at the
  open of bar i+latency_bars, never inside the same bar (no lookahead)
- partial fills: `fill_ratio` scales down the size actually obtained

Slippage is always applied in the adverse direction: entries fill worse
than the ideal price, exits fill worse than the ideal price.

## Why liquidity role is a parameter

`execution/executor.py` quotes maker-first by design — limit at the near
touch, post-only where supported, crossing only for the unfilled remainder
after a timeout, because "scalping lives and dies on fees". A cost model
that charges taker on every fill therefore prices a strategy the system
does not run, and at these horizons the gap decides whether a signal clears
its floor at all: ~0.4% round trip taker versus ~0.04% maker, against a
measured feature spread near 0.6%.

## Maker fills are not free, and this is where backtests lie

A resting order is a free option written to the market. It fills when
someone wants to trade against it — which is disproportionately when they
know something you do not. You miss the move you called correctly and get
filled on the one you called wrong. That is adverse selection, and it is
the true cost of quoting.

`maker_fill_probability` and `maker_adverse_selection` exist so that cost
is stated rather than assumed away. Leaving them at their defaults (fills
always, costs nothing) makes maker backtests optimistic in exactly the way
taker-only ones are pessimistic. Neither default is the truth; both are
knobs you are expected to justify with measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

HIGH_VOL_REGIME = "high"

MAKER = "maker"
TAKER = "taker"


@dataclass(frozen=True)
class CostModel:
    taker_fee: float = 0.001          # 0.1% — conservative end of Bybit's range
    maker_fee: float | None = None    # None -> maker pays taker rate (spot-like)
    slippage_base: float = 0.001      # 0.1%
    slippage_high_vol: float = 0.005  # 0.5%
    latency_bars: int = 1
    fill_ratio: float = 1.0           # fraction of intended size actually filled

    # Maker-specific. Defaults describe a maker who always fills and is
    # never adversely selected -- deliberately optimistic, see module docs.
    maker_fill_probability: float = 1.0
    maker_adverse_selection: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.fill_ratio <= 1:
            raise ValueError("fill_ratio must be in (0, 1]")
        if self.latency_bars < 1:
            raise ValueError("latency_bars must be >= 1 (same-bar fills are lookahead)")
        if self.maker_fee is not None and self.maker_fee < 0:
            raise ValueError("maker_fee must be >= 0 (rebates are not modelled)")
        if not 0 < self.maker_fill_probability <= 1:
            raise ValueError("maker_fill_probability must be in (0, 1]")
        if self.maker_adverse_selection < 0:
            raise ValueError("maker_adverse_selection must be >= 0 (it is a cost)")

    # ---- fees -------------------------------------------------------------

    def fee_rate(self, role: str = TAKER) -> float:
        """Fee rate for a fill in the given liquidity role."""
        if role == MAKER:
            return self.taker_fee if self.maker_fee is None else self.maker_fee
        if role == TAKER:
            return self.taker_fee
        raise ValueError(f"unknown liquidity role: {role!r}")

    def fee(self, notional: float, role: str = TAKER) -> float:
        return abs(notional) * self.fee_rate(role)

    # ---- slippage ---------------------------------------------------------

    def slippage_rate(self, volatility_regime: object, role: str = TAKER) -> float:
        """Adverse price movement on a fill.

        A taker crosses the spread and pays the regime slippage. A maker set
        the price, so it pays no spread -- but is adversely selected instead,
        which is a distinct and separately-configured cost.
        """
        if role == MAKER:
            return self.maker_adverse_selection
        if volatility_regime == HIGH_VOL_REGIME:
            return self.slippage_high_vol
        return self.slippage_base

    def entry_price(
        self, ideal_price: float, direction: int, volatility_regime: object, role: str = TAKER
    ) -> float:
        """Adverse fill on entry: longs buy higher, shorts sell lower."""
        slip = self.slippage_rate(volatility_regime, role)
        return ideal_price * (1 + direction * slip)

    def exit_price(
        self, ideal_price: float, direction: int, volatility_regime: object, role: str = TAKER
    ) -> float:
        """Adverse fill on exit: longs sell lower, shorts buy back higher."""
        slip = self.slippage_rate(volatility_regime, role)
        return ideal_price * (1 - direction * slip)

    # ---- round trip -------------------------------------------------------

    def round_trip_cost(self, volatility_regime: object, role: str = TAKER) -> float:
        """Fees + slippage on both legs of a trade, as a fraction of price.

        This is the floor a signal has to clear to be worth trading.
        """
        return 2 * self.fee_rate(role) + 2 * self.slippage_rate(volatility_regime, role)

    def effective_round_trip_cost(self, volatility_regime: object) -> float:
        """Round trip for the executor's actual behaviour: quote as maker,
        cross for whatever does not fill.

        With `maker_fill_probability` at its default of 1.0 this equals the
        pure maker cost. Lower it to the rate you actually measure and the
        taker fallback is blended back in.
        """
        p = self.maker_fill_probability
        return (
            p * self.round_trip_cost(volatility_regime, MAKER)
            + (1 - p) * self.round_trip_cost(volatility_regime, TAKER)
        )
