"""Thin wrapper around CCXT's Bybit adapter.

Centralizing exchange construction here means every other module (data
downloader now, execution module in a later milestone) gets identical
testnet/live handling, rate limiting, and retry behavior for free.
"""
from __future__ import annotations

import logging
from typing import Any

import ccxt
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cognition.config import AppConfig, Secrets
from cognition.utils.logging import get_logger

logger = get_logger("bybit_client")

RETRYABLE_EXCEPTIONS = (
    ccxt.NetworkError,
    ccxt.ExchangeNotAvailable,
    ccxt.RequestTimeout,
    ccxt.DDoSProtection,
)


class BybitClient:
    """Wraps ccxt.bybit. Testnet by default — `secrets.bybit_env` must be
    explicitly "live" (plus the ALLOW_LIVE_TRADING guard, enforced in
    config.py) before this ever talks to the real exchange.
    """

    def __init__(self, config: AppConfig, secrets: Secrets):
        self.config = config
        self.secrets = secrets
        self._exchange: ccxt.bybit | None = None

    @property
    def is_testnet(self) -> bool:
        return self.secrets.bybit_env == "testnet"

    @property
    def exchange(self) -> ccxt.bybit:
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.config.exchange.id)
            params: dict[str, Any] = {
                "enableRateLimit": True,
                "options": {"defaultType": self.config.exchange.category},
            }
            if self.secrets.bybit_api_key:
                params["apiKey"] = self.secrets.bybit_api_key
                params["secret"] = self.secrets.bybit_api_secret

            instance = exchange_class(params)
            if self.is_testnet:
                instance.set_sandbox_mode(True)
            self._exchange = instance
            logger.info(
                "Bybit client initialized",
                extra={"extra_fields": {"testnet": self.is_testnet, "authenticated": bool(self.secrets.bybit_api_key)}},
            )
        return self._exchange

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: int | None = None,
        limit: int = 1000,
    ) -> list[list[float]]:
        """Fetch one page of OHLCV candles. Raises on non-retryable
        exchange errors (e.g. bad symbol) so the caller can react.
        """
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    def milliseconds(self) -> int:
        return self.exchange.milliseconds()

    def parse_timeframe_ms(self, timeframe: str) -> int:
        return self.exchange.parse_timeframe(timeframe) * 1000

    # ---- order / market-data methods (Milestone 5+) -------------------
    # Order placement is NOT retried automatically: a timed-out create
    # might still have reached the exchange, and blind retries can double
    # an order. Callers must reconcile via fetch_order instead.

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return self.exchange.fetch_order(order_id, symbol)

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.exchange.create_order(symbol, order_type, side, amount, price, params or {})

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        return self.exchange.cancel_order(order_id, symbol)
