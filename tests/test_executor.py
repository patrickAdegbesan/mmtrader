"""Limit-first executor tests against a stateful fake exchange."""
from unittest.mock import MagicMock

import pytest

from cognition.config import ExecutionConfig
from cognition.execution.executor import LimitFirstExecutor, OrderRequest


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class FakeExchangeClient:
    """State machine standing in for BybitClient:
    - the limit order reports (limit_status, limit_filled, limit_avg)
      until cancel_order is called, then (post_cancel_status, same fill);
    - a market order always fills its full requested amount at market_avg.
    """

    def __init__(
        self,
        bid=100.0,
        ask=100.1,
        limit_status="open",
        limit_filled=0.0,
        limit_avg=0.0,
        post_cancel_status="canceled",
        post_cancel_filled=None,
        market_avg=100.2,
        cancel_raises=False,
    ):
        self.is_testnet = True
        self._ticker = {"bid": bid, "ask": ask, "last": (bid + ask) / 2}
        self.limit_status = limit_status
        self.limit_filled = limit_filled
        self.limit_avg = limit_avg
        self.post_cancel_status = post_cancel_status
        self.post_cancel_filled = post_cancel_filled if post_cancel_filled is not None else limit_filled
        self.market_avg = market_avg
        self.cancel_raises = cancel_raises

        self.canceled = False
        self.cancel_calls = []
        self.created_orders = []          # (symbol, type, side, amount, price, params)
        self._market_amounts = {}

    def fetch_ticker(self, symbol):
        return self._ticker

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        self.created_orders.append((symbol, order_type, side, amount, price, params or {}))
        order_id = f"{order_type}-{len(self.created_orders)}"
        if order_type == "market":
            self._market_amounts[order_id] = amount
        return {"id": order_id}

    def fetch_order(self, order_id, symbol):
        if order_id in self._market_amounts:
            return {"status": "closed", "filled": self._market_amounts[order_id], "average": self.market_avg}
        if self.canceled:
            return {"status": self.post_cancel_status, "filled": self.post_cancel_filled, "average": self.limit_avg}
        return {"status": self.limit_status, "filled": self.limit_filled, "average": self.limit_avg}

    def cancel_order(self, order_id, symbol):
        self.cancel_calls.append(order_id)
        if self.cancel_raises:
            raise Exception("order already filled")
        self.canceled = True
        return {"id": order_id}


def make_executor(client, **config_overrides):
    config = ExecutionConfig(**config_overrides)
    clock = FakeClock()
    ex = LimitFirstExecutor.__new__(LimitFirstExecutor)  # bypass live-guard for fakes
    ex.client = client
    ex.config = config
    ex._clock = clock
    ex._sleep = clock.sleep
    return ex


def test_buy_quotes_at_the_bid_and_sell_at_the_ask():
    buy_client = FakeExchangeClient(limit_status="closed", limit_filled=1.0, limit_avg=100.0)
    make_executor(buy_client).execute(OrderRequest("BTC/USDT", 1, 1.0))
    _, order_type, side, _, price, _ = buy_client.created_orders[0]
    assert (order_type, side, price) == ("limit", "buy", 100.0)   # joins the bid

    sell_client = FakeExchangeClient(limit_status="closed", limit_filled=1.0, limit_avg=100.1)
    make_executor(sell_client).execute(OrderRequest("BTC/USDT", -1, 1.0))
    _, order_type, side, _, price, _ = sell_client.created_orders[0]
    assert (order_type, side, price) == ("limit", "sell", 100.1)  # joins the ask


def test_immediate_fill_never_touches_market_fallback():
    client = FakeExchangeClient(limit_status="closed", limit_filled=2.0, limit_avg=100.0)
    result = make_executor(client).execute(OrderRequest("BTC/USDT", 1, 2.0))
    assert result.fully_filled
    assert not result.used_market_fallback
    assert len(client.created_orders) == 1
    assert client.cancel_calls == []


def test_timeout_cancels_and_falls_back_to_market():
    client = FakeExchangeClient(limit_status="open", limit_filled=0.0, market_avg=100.2)
    ex = make_executor(client, limit_timeout_seconds=3.0, poll_interval_seconds=0.5)
    result = ex.execute(OrderRequest("BTC/USDT", 1, 2.0))
    assert result.used_market_fallback
    assert result.fully_filled
    assert result.average_price == pytest.approx(100.2)
    assert client.cancel_calls == ["limit-1"]
    _, order_type, _, amount, _, _ = client.created_orders[1]
    assert order_type == "market" and amount == pytest.approx(2.0)


def test_partial_fill_only_markets_the_remainder_and_blends_price():
    client = FakeExchangeClient(limit_status="open", limit_filled=1.5, limit_avg=100.0, market_avg=100.4)
    ex = make_executor(client, limit_timeout_seconds=2.0, poll_interval_seconds=0.5)
    result = ex.execute(OrderRequest("BTC/USDT", 1, 2.0))
    assert result.fully_filled
    _, order_type, _, amount, _, _ = client.created_orders[1]
    assert order_type == "market" and amount == pytest.approx(0.5)
    assert result.average_price == pytest.approx((100.0 * 1.5 + 100.4 * 0.5) / 2.0)


def test_no_market_fallback_when_disabled():
    client = FakeExchangeClient(limit_status="open", limit_filled=0.5, limit_avg=100.0)
    ex = make_executor(client, limit_timeout_seconds=1.0, poll_interval_seconds=0.5, fallback_to_market=False)
    result = ex.execute(OrderRequest("BTC/USDT", 1, 2.0))
    assert not result.used_market_fallback
    assert result.filled_quantity == pytest.approx(0.5)
    assert len(client.created_orders) == 1


def test_cancel_race_with_fill_is_reconciled_without_market_order():
    # The order fills completely just as we try to cancel: cancel raises,
    # reconciliation sees the full fill, and no market order is sent.
    client = FakeExchangeClient(
        limit_status="open", limit_filled=0.0, limit_avg=0.0, cancel_raises=True,
    )

    def fetch_order_race(order_id, symbol):
        if client.cancel_calls:  # after the cancel attempt: it had filled
            return {"status": "closed", "filled": 2.0, "average": 100.0}
        return {"status": "open", "filled": 0.0, "average": 0.0}

    client.fetch_order = fetch_order_race
    ex = make_executor(client, limit_timeout_seconds=2.0, poll_interval_seconds=0.5)
    result = ex.execute(OrderRequest("BTC/USDT", 1, 2.0))
    assert result.fully_filled
    assert not result.used_market_fallback
    assert len(client.created_orders) == 1


def test_post_only_param_is_sent():
    client = FakeExchangeClient(limit_status="closed", limit_filled=1.0, limit_avg=100.0)
    make_executor(client, post_only=True).execute(OrderRequest("BTC/USDT", 1, 1.0))
    params = client.created_orders[0][5]
    assert params == {"postOnly": True}


def test_executor_refuses_live_client_without_override():
    client = MagicMock()
    client.is_testnet = False
    client.secrets.allow_live_trading = ""
    with pytest.raises(RuntimeError, match="Testnet only"):
        LimitFirstExecutor(client, ExecutionConfig())


def test_executor_accepts_testnet_client():
    client = MagicMock()
    client.is_testnet = True
    LimitFirstExecutor(client, ExecutionConfig())  # no raise
