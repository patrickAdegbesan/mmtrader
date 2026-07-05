"""Limit-first order execution (Milestone 5).

Strategy: quote as a maker first (limit at the near touch, post-only
where supported), poll for the fill, and only cross the spread with a
market order for the unfilled remainder after a timeout — scalping
lives and dies on fees, so we pay taker only when we must.

Safety:
- Refuses to construct against a live (non-testnet) client unless the
  environment-level live-trading override is present. This is the second
  independent live-trading lock (the first is in config.py).
- Order placement itself is never blind-retried (a timeout may mean the
  order DID reach the exchange); reconciliation goes through fetch_order.

`clock` and `sleep` are injectable for deterministic tests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from cognition.config import ExecutionConfig
from cognition.data.bybit_client import BybitClient
from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("executor")

FILLED_STATUSES = {"closed", "filled"}
OPEN_STATUSES = {"open", "new", "partially_filled"}


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    direction: int              # 1 buy, -1 sell
    quantity: float

    @property
    def side(self) -> str:
        return "buy" if self.direction == 1 else "sell"


@dataclass
class ExecutionResult:
    request: OrderRequest
    filled_quantity: float = 0.0
    average_price: float = 0.0
    used_market_fallback: bool = False
    order_ids: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def fully_filled(self) -> bool:
        return self.filled_quantity >= self.request.quantity * 0.999


class LimitFirstExecutor:
    def __init__(
        self,
        client: BybitClient,
        config: ExecutionConfig,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not client.is_testnet and client.secrets.allow_live_trading != "I_UNDERSTAND_THE_RISK":
            raise RuntimeError(
                "LimitFirstExecutor refuses to run against live Bybit without the "
                "explicit ALLOW_LIVE_TRADING override. Testnet only in this phase."
            )
        self.client = client
        self.config = config
        self._clock = clock
        self._sleep = sleep

    def _maker_price(self, symbol: str, direction: int) -> float:
        ticker = self.client.fetch_ticker(symbol)
        bid, ask = ticker.get("bid"), ticker.get("ask")
        if not bid or not ask:
            last = ticker.get("last")
            if not last:
                raise RuntimeError(f"No usable prices in ticker for {symbol}")
            return float(last)
        # Join the near touch: buyers quote at the bid, sellers at the ask.
        return float(bid if direction == 1 else ask)

    def execute(self, request: OrderRequest) -> ExecutionResult:
        started = self._clock()
        result = ExecutionResult(request=request)
        price = self._maker_price(request.symbol, request.direction)

        params = {"postOnly": True} if self.config.post_only else {}
        order = self.client.create_order(
            request.symbol, "limit", request.side, request.quantity, price, params
        )
        order_id = order["id"]
        result.order_ids.append(order_id)
        log_with_fields(
            logger, 20, "Limit order placed",
            symbol=request.symbol, side=request.side, quantity=request.quantity, price=price, order_id=order_id,
        )

        filled, avg = self._await_fill(order_id, request.symbol)
        result.filled_quantity = filled
        result.average_price = avg

        remaining = request.quantity - filled
        if remaining > request.quantity * 0.001 and self.config.fallback_to_market:
            self._cancel_quietly(order_id, request.symbol)
            # Re-check: the cancel may have raced a fill.
            filled, avg = self._reconcile(order_id, request.symbol)
            result.filled_quantity = filled
            result.average_price = avg
            remaining = request.quantity - filled
            if remaining > request.quantity * 0.001:
                market = self.client.create_order(request.symbol, "market", request.side, remaining)
                result.order_ids.append(market["id"])
                result.used_market_fallback = True
                m_filled, m_avg = self._reconcile(market["id"], request.symbol)
                total = result.filled_quantity + m_filled
                if total > 0:
                    result.average_price = (
                        result.average_price * result.filled_quantity + m_avg * m_filled
                    ) / total
                result.filled_quantity = total
                log_with_fields(
                    logger, 30, "Market fallback used",
                    symbol=request.symbol, remaining=remaining, market_order_id=market["id"],
                )

        result.elapsed_seconds = self._clock() - started
        log_with_fields(
            logger, 20, "Execution complete",
            symbol=request.symbol, side=request.side,
            filled=result.filled_quantity, avg_price=result.average_price,
            market_fallback=result.used_market_fallback, elapsed=round(result.elapsed_seconds, 3),
        )
        return result

    def _await_fill(self, order_id: str, symbol: str) -> tuple[float, float]:
        deadline = self._clock() + self.config.limit_timeout_seconds
        filled, avg = 0.0, 0.0
        while self._clock() < deadline:
            order = self.client.fetch_order(order_id, symbol)
            filled = float(order.get("filled") or 0.0)
            avg = float(order.get("average") or 0.0)
            if (order.get("status") or "").lower() in FILLED_STATUSES:
                return filled, avg
            self._sleep(self.config.poll_interval_seconds)
        return filled, avg

    def _reconcile(self, order_id: str, symbol: str) -> tuple[float, float]:
        order = self.client.fetch_order(order_id, symbol)
        return float(order.get("filled") or 0.0), float(order.get("average") or 0.0)

    def _cancel_quietly(self, order_id: str, symbol: str) -> None:
        try:
            self.client.cancel_order(order_id, symbol)
        except Exception as exc:  # already filled/canceled is fine — reconcile decides
            log_with_fields(logger, 30, "Cancel failed (may already be filled)", order_id=order_id, error=str(exc))
