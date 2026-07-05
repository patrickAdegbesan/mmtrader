"""Guardian A — the Risk Engine.

Standalone module. EVERY proposed trade passes through evaluate() BEFORE
execution; agents can never override it. Implements the project's
non-negotiable rules:

1. Risk per trade: hard cap 1% of equity, fractional Kelly sizing
   within that cap.
2. Daily circuit breaker: all trading halts at -5% daily equity.
3. Drawdown governor: position sizes shrink automatically while total
   drawdown from peak exceeds its threshold.
4. Overtrading guard: max trades per hour / per day.
5. Exposure limits: max concurrent positions and max total notional
   exposure across symbols.
6. Stop-loss / take-profit clamping: model outputs are clamped to hard
   bounds — a proposal without a stop is rejected outright.

The corresponding config validators (config.py) refuse values looser
than the non-negotiables, so the rules cannot be silently relaxed via
YAML either.

Every decision returns a full audit trail (reasons + adjustments) for
Guardian B's structured logging.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("risk_engine")

MS_PER_HOUR = 3_600_000


def _day_key(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class TradeProposal:
    symbol: str
    direction: int                     # 1 long, -1 short
    price: float                       # current reference price
    stop_loss_pct: float               # model output, will be clamped
    take_profit_pct: float
    timestamp_ms: int
    win_prob: float | None = None      # ensemble estimate, for Kelly
    reward_risk: float | None = None   # payoff ratio, for Kelly


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    quantity: float = 0.0
    notional: float = 0.0
    risk_amount: float = 0.0
    risk_fraction: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    adjustments: list[str] = field(default_factory=list)


@dataclass
class RiskLimits:
    """Mirrors RiskConfig (config.py); plain dataclass so the engine has
    no pydantic dependency and tests can construct it directly."""
    risk_per_trade: float = 0.01
    kelly_fraction: float = 0.5
    daily_loss_halt: float = 0.05
    drawdown_governor_threshold: float = 0.10
    drawdown_governor_scale: float = 0.5
    max_trades_per_hour: int = 6
    max_trades_per_day: int = 30
    max_concurrent_positions: int = 2
    max_total_exposure: float = 1.0    # fraction of equity
    min_stop_pct: float = 0.001
    max_stop_pct: float = 0.02
    min_take_profit_pct: float = 0.001
    max_take_profit_pct: float = 0.05


class RiskEngine:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._trade_times: deque[int] = deque()
        self._open_positions: dict[str, float] = {}   # symbol -> notional
        self._day: str | None = None
        self._day_start_equity: float | None = None
        self._peak_equity: float = 0.0
        self._halted_day: str | None = None

    # ---- state the caller feeds in -----------------------------------

    def _roll_day(self, timestamp_ms: int, equity: float) -> None:
        day = _day_key(timestamp_ms)
        if day != self._day:
            self._day = day
            self._day_start_equity = equity
            if self._halted_day is not None and self._halted_day != day:
                log_with_fields(logger, 30, "Circuit breaker reset for new day", day=day)
                self._halted_day = None

    def record_open(self, symbol: str, notional: float, timestamp_ms: int) -> None:
        self._open_positions[symbol] = self._open_positions.get(symbol, 0.0) + notional
        self._trade_times.append(timestamp_ms)

    def record_close(self, symbol: str, equity_after: float) -> None:
        self._open_positions.pop(symbol, None)
        self._peak_equity = max(self._peak_equity, equity_after)

    @property
    def is_halted(self) -> bool:
        return self._halted_day is not None and self._halted_day == self._day

    @property
    def open_position_count(self) -> int:
        return len(self._open_positions)

    @property
    def total_exposure(self) -> float:
        return sum(self._open_positions.values())

    # ---- the gate -----------------------------------------------------

    def evaluate(self, proposal: TradeProposal, equity: float) -> RiskDecision:
        limits = self.limits
        adjustments: list[str] = []
        self._roll_day(proposal.timestamp_ms, equity)
        self._peak_equity = max(self._peak_equity, equity)

        def reject(reason: str) -> RiskDecision:
            decision = RiskDecision(approved=False, reason=reason, adjustments=adjustments)
            log_with_fields(
                logger, 30, "Trade REJECTED",
                symbol=proposal.symbol, direction=proposal.direction, reason=reason,
            )
            return decision

        # --- structural sanity ---
        if proposal.direction not in (1, -1):
            return reject("invalid direction")
        if proposal.stop_loss_pct <= 0:
            return reject("proposal has no stop-loss")
        if equity <= 0:
            return reject("non-positive equity")

        # --- rule 2: daily circuit breaker ---
        daily_loss = (equity - self._day_start_equity) / self._day_start_equity
        if self.is_halted:
            return reject("daily circuit breaker active (review mode until next UTC day)")
        if daily_loss <= -limits.daily_loss_halt:
            self._halted_day = self._day
            log_with_fields(
                logger, 40, "CIRCUIT BREAKER TRIPPED",
                daily_loss=round(daily_loss, 4), day=self._day,
            )
            return reject(f"daily circuit breaker tripped at {daily_loss:.2%}")

        # --- rule 4: overtrading guard ---
        now = proposal.timestamp_ms
        while self._trade_times and self._trade_times[0] < now - 24 * MS_PER_HOUR:
            self._trade_times.popleft()
        day_key = _day_key(now)
        trades_today = sum(1 for t in self._trade_times if _day_key(t) == day_key)
        trades_last_hour = sum(1 for t in self._trade_times if t >= now - MS_PER_HOUR)
        if trades_last_hour >= limits.max_trades_per_hour:
            return reject(f"overtrading guard: {trades_last_hour} trades in the last hour")
        if trades_today >= limits.max_trades_per_day:
            return reject(f"overtrading guard: {trades_today} trades today")

        # --- rule 5: exposure limits ---
        if self.open_position_count >= limits.max_concurrent_positions:
            return reject(f"exposure: {self.open_position_count} concurrent positions open")
        if proposal.symbol in self._open_positions:
            return reject(f"exposure: position already open in {proposal.symbol}")

        # --- rule 6: clamp model-proposed SL/TP ---
        stop = min(max(proposal.stop_loss_pct, limits.min_stop_pct), limits.max_stop_pct)
        if stop != proposal.stop_loss_pct:
            adjustments.append(f"stop clamped {proposal.stop_loss_pct:.4f} -> {stop:.4f}")
        tp = min(max(proposal.take_profit_pct, limits.min_take_profit_pct), limits.max_take_profit_pct)
        if tp != proposal.take_profit_pct:
            adjustments.append(f"take-profit clamped {proposal.take_profit_pct:.4f} -> {tp:.4f}")

        # --- rule 1: fractional Kelly inside the 1% hard cap ---
        risk_fraction = limits.risk_per_trade
        if proposal.win_prob is not None:
            rr = proposal.reward_risk if proposal.reward_risk else tp / stop
            kelly = proposal.win_prob - (1 - proposal.win_prob) / rr
            if kelly <= 0:
                return reject(f"no edge: kelly={kelly:.4f} (win_prob={proposal.win_prob:.2f}, rr={rr:.2f})")
            scaled = limits.kelly_fraction * kelly
            if scaled < risk_fraction:
                adjustments.append(f"kelly sizing {scaled:.4%} below cap {risk_fraction:.2%}")
            risk_fraction = min(risk_fraction, scaled)

        # --- rule 3: drawdown governor ---
        drawdown = 1 - equity / self._peak_equity if self._peak_equity > 0 else 0.0
        if drawdown > limits.drawdown_governor_threshold:
            risk_fraction *= limits.drawdown_governor_scale
            adjustments.append(
                f"drawdown governor: dd={drawdown:.2%} > {limits.drawdown_governor_threshold:.0%}, "
                f"risk scaled by {limits.drawdown_governor_scale}"
            )

        # --- sizing ---
        risk_amount = equity * risk_fraction
        stop_distance = proposal.price * stop
        quantity = risk_amount / stop_distance
        notional = quantity * proposal.price

        # rule 5 continued: total exposure cap (shrink, then reject if hopeless)
        exposure_room = equity * limits.max_total_exposure - self.total_exposure
        if exposure_room <= 0:
            return reject("exposure: total exposure limit reached")
        if notional > exposure_room:
            adjustments.append(f"notional shrunk {notional:.2f} -> {exposure_room:.2f} (exposure cap)")
            quantity *= exposure_room / notional
            notional = exposure_room
            risk_amount = quantity * stop_distance

        decision = RiskDecision(
            approved=True, reason="approved",
            quantity=quantity, notional=notional,
            risk_amount=risk_amount, risk_fraction=risk_amount / equity,
            stop_loss_pct=stop, take_profit_pct=tp,
            adjustments=adjustments,
        )
        log_with_fields(
            logger, 20, "Trade approved",
            symbol=proposal.symbol, direction=proposal.direction,
            quantity=round(quantity, 8), notional=round(notional, 2),
            risk_fraction=round(decision.risk_fraction, 5), adjustments=adjustments,
        )
        return decision
