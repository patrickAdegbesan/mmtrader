"""Bybit v5 public WebSocket kline feed with automatic reconnection.

Runs an asyncio consumer on a background thread and hands CLOSED candles
(payload `confirm: true`) to the synchronous trading loop through a
queue. Reconnects with exponential backoff on any disconnect and
re-subscribes; a heartbeat ping keeps the connection alive.

The connection factory is injectable, so reconnection and parsing logic
are fully unit-testable without a network. NOTE: some proxied
environments (including the Claude Code sandbox this was written in)
cannot pass WebSocket upgrades at all — RestPollingFeed is the fallback
that works over plain HTTPS.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Awaitable, Callable

from cognition.data.feed import BarEvent
from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("ws_feed")

MAINNET_URL = "wss://stream.bybit.com/v5/public/spot"
TESTNET_URL = "wss://stream-testnet.bybit.com/v5/public/spot"

_INTERVAL_MAP = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}


def kline_topic(symbol: str, timeframe: str) -> str:
    return f"kline.{_INTERVAL_MAP[timeframe]}.{symbol.replace('/', '')}"


def parse_kline_message(raw: str, topic_to_symbol: dict[str, str]) -> list[BarEvent]:
    """Extract CLOSED candles from a Bybit v5 kline message. Unconfirmed
    (still-forming) candles are ignored."""
    message = json.loads(raw)
    topic = message.get("topic", "")
    symbol = topic_to_symbol.get(topic)
    if symbol is None:
        return []
    events = []
    for item in message.get("data", []):
        if not item.get("confirm"):
            continue
        events.append(BarEvent(
            symbol=symbol,
            timestamp=int(item["start"]),
            open=float(item["open"]), high=float(item["high"]),
            low=float(item["low"]), close=float(item["close"]),
            volume=float(item["volume"]),
        ))
    return events


class BybitWebSocketFeed:
    def __init__(
        self,
        symbols: list[str],
        timeframe: str = "1m",
        testnet: bool = True,
        connect: Callable[[str], Awaitable] | None = None,
        max_backoff_seconds: float = 60.0,
        ping_interval_seconds: float = 20.0,
    ):
        self.url = TESTNET_URL if testnet else MAINNET_URL
        self.topic_to_symbol = {kline_topic(s, timeframe): s for s in symbols}
        self._connect = connect or self._default_connect
        self.max_backoff = max_backoff_seconds
        self.ping_interval = ping_interval_seconds
        self.queue: queue.Queue[BarEvent] = queue.Queue()
        self.reconnect_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    async def _default_connect(url: str):
        import websockets  # imported lazily: optional dependency
        return await websockets.connect(url, ping_interval=None)

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_thread, daemon=True, name="ws-feed")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run_thread(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                ws = await self._connect(self.url)
                await ws.send(json.dumps({"op": "subscribe", "args": list(self.topic_to_symbol)}))
                log_with_fields(logger, 20, "WebSocket subscribed", topics=list(self.topic_to_symbol))
                backoff = 1.0
                await self._consume(ws)
            except Exception as exc:
                if self._stop.is_set():
                    break
                self.reconnect_count += 1
                log_with_fields(
                    logger, 40, "WebSocket disconnected — reconnecting",
                    error=str(exc), backoff_seconds=backoff, reconnects=self.reconnect_count,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

    async def _consume(self, ws) -> None:
        last_ping = asyncio.get_event_loop().time()
        while not self._stop.is_set():
            now = asyncio.get_event_loop().time()
            if now - last_ping >= self.ping_interval:
                await ws.send(json.dumps({"op": "ping"}))
                last_ping = now
            raw = await asyncio.wait_for(ws.recv(), timeout=self.ping_interval)
            for event in parse_kline_message(raw, self.topic_to_symbol):
                self.queue.put(event)

    # ---- consumption from the sync loop ---------------------------------

    def bars(self):
        """Blocking iterator over closed candles (call start() first)."""
        while not self._stop.is_set():
            try:
                yield self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
