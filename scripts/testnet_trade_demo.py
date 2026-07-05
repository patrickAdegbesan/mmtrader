#!/usr/bin/env python3
"""Milestone 5 demo: the full pre-trade pipeline —
proposal -> Guardian A (Risk Engine) -> limit-first executor.

Modes:
    python scripts/testnet_trade_demo.py                # dry-run (mock exchange)
    python scripts/testnet_trade_demo.py --testnet      # real Bybit TESTNET order
                                                        # (needs testnet keys in .env
                                                        # and network access to Bybit)

The dry-run mode uses a scripted mock exchange so the entire flow —
including risk rejection paths and the market fallback — can be
demonstrated without network access. NEVER runs against live Bybit:
the executor refuses non-testnet clients outright.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cognition.config import get_settings  # noqa: E402
from cognition.data.bybit_client import BybitClient  # noqa: E402
from cognition.execution.executor import LimitFirstExecutor, OrderRequest  # noqa: E402
from cognition.risk.engine import RiskEngine, RiskLimits, TradeProposal  # noqa: E402
from cognition.utils.logging import get_logger, log_with_fields  # noqa: E402


def make_mock_client() -> MagicMock:
    """Stateful scripted exchange: the limit order partially fills, then
    the market fallback completes the remainder."""
    client = MagicMock()
    client.is_testnet = True
    client.fetch_ticker.return_value = {"bid": 50_000.0, "ask": 50_001.0, "last": 50_000.5}

    state = {"canceled": False, "market_amounts": {}}

    def create_order(symbol, order_type, side, amount, price=None, params=None):
        order_id = f"{order_type}-1"
        if order_type == "market":
            state["market_amounts"][order_id] = amount
        return {"id": order_id}

    def fetch_order(order_id, symbol):
        if order_id in state["market_amounts"]:
            return {"status": "closed", "filled": state["market_amounts"][order_id], "average": 50_001.0}
        status = "canceled" if state["canceled"] else "open"
        return {"status": status, "filled": 0.05, "average": 50_000.0}  # partial limit fill

    def cancel_order(order_id, symbol):
        state["canceled"] = True
        return {"id": order_id}

    client.create_order.side_effect = create_order
    client.fetch_order.side_effect = fetch_order
    client.cancel_order.side_effect = cancel_order
    return client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testnet", action="store_true", help="Place a real order on Bybit TESTNET")
    parser.add_argument("--equity", type=float, default=10_000.0)
    args = parser.parse_args()

    config, secrets = get_settings()
    logger = get_logger("trade_demo", log_dir=config.log_dir)

    limits = RiskLimits(**config.risk.model_dump())
    risk_engine = RiskEngine(limits)
    now_ms = int(time.time() * 1000)

    print("=== Guardian A demonstration ===\n")

    # 1. A sane proposal (as the ensemble would emit) — should pass.
    proposal = TradeProposal(
        symbol="BTC/USDT", direction=1, price=50_000.0,
        stop_loss_pct=0.004, take_profit_pct=0.006,
        timestamp_ms=now_ms, win_prob=0.6, reward_risk=1.5,
    )
    decision = risk_engine.evaluate(proposal, equity=args.equity)
    print(f"[1] Sane proposal      -> approved={decision.approved}  qty={decision.quantity:.6f} "
          f"risk={decision.risk_fraction:.3%}  adjustments={decision.adjustments}")

    # 2. A reckless stop (5%) — must be clamped to the hard bound.
    reckless = TradeProposal(
        symbol="ETH/USDT", direction=-1, price=3_000.0,
        stop_loss_pct=0.05, take_profit_pct=0.10, timestamp_ms=now_ms,
    )
    d2 = risk_engine.evaluate(reckless, equity=args.equity)
    print(f"[2] Reckless 5% stop   -> approved={d2.approved}  stop clamped to {d2.stop_loss_pct:.3%}  "
          f"adjustments={d2.adjustments}")

    # 3. No edge (Kelly <= 0) — must be rejected.
    no_edge = TradeProposal(
        symbol="BTC/USDT", direction=1, price=50_000.0,
        stop_loss_pct=0.004, take_profit_pct=0.004,
        timestamp_ms=now_ms, win_prob=0.4, reward_risk=1.0,
    )
    d3 = risk_engine.evaluate(no_edge, equity=args.equity)
    print(f"[3] No-edge proposal   -> approved={d3.approved}  reason: {d3.reason}")

    # 4. Daily circuit breaker: equity down 6% today — halt everything.
    breaker_engine = RiskEngine(limits)
    breaker_engine.evaluate(proposal, equity=10_000.0)          # anchors day-start equity
    d4 = breaker_engine.evaluate(proposal, equity=9_400.0)      # -6% intraday
    d5 = breaker_engine.evaluate(proposal, equity=9_800.0)      # even after recovery: halted
    print(f"[4] -6% intraday       -> approved={d4.approved}  reason: {d4.reason}")
    print(f"[5] Post-halt attempt  -> approved={d5.approved}  reason: {d5.reason}")

    # --- Execution --------------------------------------------------------
    print("\n=== Limit-first execution ===\n")
    if args.testnet:
        client = BybitClient(config, secrets)
        if not client.is_testnet:
            print("Refusing: BYBIT_ENV is not testnet.")
            return 1
        executor = LimitFirstExecutor(client, config.execution)
    else:
        client = make_mock_client()
        client.secrets = secrets
        fake_time = iter(range(0, 100))
        executor = LimitFirstExecutor.__new__(LimitFirstExecutor)
        executor.client = client
        executor.config = config.execution
        executor._clock = lambda: next(fake_time)
        executor._sleep = lambda s: None
        print("(dry-run: scripted mock exchange — limit half-fills, market finishes)\n")

    if decision.approved:
        request = OrderRequest(symbol=proposal.symbol, direction=proposal.direction, quantity=round(decision.quantity, 6))
        result = executor.execute(request)
        risk_engine.record_open(proposal.symbol, decision.notional, now_ms)
        print(f"\nExecution: filled {result.filled_quantity} @ avg {result.average_price} "
              f"(market fallback: {result.used_market_fallback}, orders: {result.order_ids})")
        log_with_fields(
            logger, 20, "Demo trade executed",
            symbol=request.symbol, filled=result.filled_quantity,
            avg_price=result.average_price, fallback=result.used_market_fallback,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
