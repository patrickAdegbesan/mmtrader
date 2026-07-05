"""Market data feeds for paper/live trading.

Three implementations behind one iterator interface:
- ReplayFeed: plays historical bars as if they were arriving live —
  the verification path in environments without exchange access, and
  useful for accelerated soak tests.
- RestPollingFeed: polls fetch_ohlcv for newly closed candles. Works
  anywhere plain HTTPS works; the dependable default for live runs.
- BybitWebSocketFeed (websocket_feed.py): push-based kline stream with
  reconnection, for lowest latency where WebSockets are available.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterator

import pandas as pd

from cognition.data.bybit_client import BybitClient
from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("feed")


@dataclass(frozen=True)
class BarEvent:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ohlcv(cls, symbol: str, row: list[float]) -> "BarEvent":
        return cls(
            symbol=symbol, timestamp=int(row[0]),
            open=float(row[1]), high=float(row[2]), low=float(row[3]),
            close=float(row[4]), volume=float(row[5]),
        )


class ReplayFeed:
    """Replays historical frames in timestamp order across symbols."""

    def __init__(self, frames: dict[str, pd.DataFrame], delay_seconds: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.frames = frames
        self.delay_seconds = delay_seconds
        self._sleep = sleep

    def bars(self) -> Iterator[BarEvent]:
        events: list[BarEvent] = []
        for symbol, df in self.frames.items():
            for row in df[["timestamp", "open", "high", "low", "close", "volume"]].itertuples(index=False):
                events.append(BarEvent(symbol, int(row.timestamp), row.open, row.high, row.low, row.close, row.volume))
        events.sort(key=lambda e: (e.timestamp, e.symbol))
        for event in events:
            if self.delay_seconds:
                self._sleep(self.delay_seconds)
            yield event


class RestPollingFeed:
    """Emits each candle once it has CLOSED (never the forming candle —
    trading on incomplete bars is lookahead's ugly cousin)."""

    def __init__(
        self,
        client: BybitClient,
        symbols: list[str],
        timeframe: str = "1m",
        poll_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock_ms: Callable[[], int] | None = None,
    ):
        self.client = client
        self.symbols = symbols
        self.timeframe = timeframe
        self.poll_seconds = poll_seconds
        self._sleep = sleep
        self._clock_ms = clock_ms or client.milliseconds
        self._last_emitted: dict[str, int] = {}
        self.running = True

    def stop(self) -> None:
        self.running = False

    def bars(self) -> Iterator[BarEvent]:
        tf_ms = self.client.parse_timeframe_ms(self.timeframe)
        while self.running:
            now = self._clock_ms()
            for symbol in self.symbols:
                try:
                    candles = self.client.fetch_ohlcv(symbol, self.timeframe, limit=3)
                except Exception as exc:
                    log_with_fields(logger, 40, "Feed fetch error", symbol=symbol, error=str(exc))
                    continue
                for row in candles:
                    ts = int(row[0])
                    is_closed = ts + tf_ms <= now
                    if is_closed and ts > self._last_emitted.get(symbol, -1):
                        self._last_emitted[symbol] = ts
                        yield BarEvent.from_ohlcv(symbol, row)
            self._sleep(self.poll_seconds)
