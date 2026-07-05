"""Feeds (replay/REST/WebSocket parsing+reconnect) and Guardian B
(alert throttling, watchdog, dashboard)."""
import asyncio
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cognition.data.feed import BarEvent, ReplayFeed, RestPollingFeed
from cognition.data.websocket_feed import BybitWebSocketFeed, kline_topic, parse_kline_message
from cognition.monitoring.alerts import FeedWatchdog, TelegramAlerter
from cognition.monitoring.dashboard import DashboardWriter, equity_svg

TF_MS = 60_000


def make_frame(symbol_offset=0, n=5, start=1_700_000_000_000):
    return pd.DataFrame({
        "timestamp": [start + i * TF_MS for i in range(n)],
        "open": 100.0 + symbol_offset, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 5.0,
    })


# ---------- replay feed ----------

def test_replay_feed_interleaves_symbols_in_time_order():
    feed = ReplayFeed({"BTC/USDT": make_frame(), "ETH/USDT": make_frame(10)})
    events = list(feed.bars())
    assert len(events) == 10
    timestamps = [e.timestamp for e in events]
    assert timestamps == sorted(timestamps)
    assert {e.symbol for e in events} == {"BTC/USDT", "ETH/USDT"}


# ---------- REST polling feed ----------

def test_rest_feed_emits_only_closed_candles_once():
    client = MagicMock()
    client.parse_timeframe_ms.return_value = TF_MS
    t0 = 1_700_000_000_000
    # Three candles; the last one (t0+2m) is still forming at "now".
    candles = [
        [t0, 100, 101, 99, 100.5, 5],
        [t0 + TF_MS, 100.5, 102, 100, 101, 6],
        [t0 + 2 * TF_MS, 101, 103, 101, 102, 7],
    ]
    client.fetch_ohlcv.return_value = candles
    now = {"ms": t0 + 2 * TF_MS + 30_000}  # 2m30s: first two candles closed

    polls = {"n": 0}
    def fake_sleep(_):
        polls["n"] += 1
        if polls["n"] >= 3:
            feed.stop()

    feed = RestPollingFeed(
        client, ["BTC/USDT"], "1m", poll_seconds=0,
        sleep=fake_sleep, clock_ms=lambda: now["ms"],
    )
    events = list(feed.bars())
    # Two closed candles, emitted exactly once despite repeated polling.
    assert [e.timestamp for e in events] == [t0, t0 + TF_MS]


def test_rest_feed_survives_fetch_errors():
    client = MagicMock()
    client.parse_timeframe_ms.return_value = TF_MS
    client.fetch_ohlcv.side_effect = Exception("rate limited")
    feed = RestPollingFeed(client, ["BTC/USDT"], "1m", poll_seconds=0,
                           sleep=lambda _: feed.stop(), clock_ms=lambda: 0)
    assert list(feed.bars()) == []  # no crash


# ---------- WebSocket feed ----------

def test_parse_kline_message_extracts_only_confirmed_candles():
    topic = kline_topic("BTC/USDT", "1m")
    raw = json.dumps({
        "topic": topic,
        "data": [
            {"start": 1_700_000_000_000, "open": "100", "high": "101", "low": "99",
             "close": "100.5", "volume": "5", "confirm": True},
            {"start": 1_700_000_060_000, "open": "100.5", "high": "102", "low": "100",
             "close": "101", "volume": "6", "confirm": False},
        ],
    })
    events = parse_kline_message(raw, {topic: "BTC/USDT"})
    assert len(events) == 1
    assert events[0] == BarEvent("BTC/USDT", 1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 5.0)


def test_parse_ignores_unknown_topics_and_control_messages():
    assert parse_kline_message(json.dumps({"op": "pong"}), {}) == []
    assert parse_kline_message(json.dumps({"topic": "other", "data": []}), {"k": "BTC"}) == []


def test_websocket_reconnects_after_disconnect():
    attempts = {"n": 0}

    class FakeWS:
        def __init__(self, fail_after):
            self.fail_after = fail_after
            self.received = 0
            self.sent = []

        async def send(self, msg):
            self.sent.append(msg)

        async def recv(self):
            self.received += 1
            if self.received > self.fail_after:
                raise ConnectionError("dropped")
            topic = kline_topic("BTC/USDT", "1m")
            return json.dumps({
                "topic": topic,
                "data": [{"start": 1_700_000_000_000 + self.received * TF_MS,
                          "open": "1", "high": "1", "low": "1", "close": "1",
                          "volume": "1", "confirm": True}],
            })

    async def fake_connect(url):
        attempts["n"] += 1
        if attempts["n"] >= 3:
            feed.stop()  # end the test after two successful reconnects
        return FakeWS(fail_after=2)

    feed = BybitWebSocketFeed(["BTC/USDT"], "1m", connect=fake_connect, max_backoff_seconds=0.01)
    asyncio.run(feed._run())
    assert attempts["n"] >= 3          # initial connect + at least 2 reconnects
    assert feed.reconnect_count >= 2
    assert feed.queue.qsize() >= 4     # bars flowed across connections


# ---------- alerter ----------

def test_alerter_sends_and_throttles_repeats():
    sent = []
    clock = {"t": 0.0}
    alerter = TelegramAlerter(
        token="tok", chat_id="chat", throttle_seconds=300,
        transport=lambda tok, chat, text: sent.append(text),
        clock=lambda: clock["t"],
    )
    assert alerter.alert("feed_stale", "gap detected") is True
    assert alerter.alert("feed_stale", "gap detected") is False   # throttled
    assert alerter.alert("feed_stale", "gap detected") is False
    clock["t"] = 301.0
    assert alerter.alert("feed_stale", "gap detected") is True
    assert "+2 similar suppressed" in sent[-1]
    assert len(sent) == 2


def test_critical_alerts_bypass_throttle():
    sent = []
    alerter = TelegramAlerter(token="tok", chat_id="chat", throttle_seconds=300,
                              transport=lambda *a: sent.append(a), clock=lambda: 0.0)
    assert alerter.alert("circuit_breaker", "halted", critical=True)
    assert alerter.alert("circuit_breaker", "halted again", critical=True)
    assert len(sent) == 2


def test_unconfigured_alerter_logs_but_does_not_crash():
    alerter = TelegramAlerter(token="", chat_id="")
    assert alerter.alert("x", "no telegram configured") is True


# ---------- watchdog ----------

def test_watchdog_detects_stale_feed():
    wd = FeedWatchdog(timeframe_ms=TF_MS, stale_multiple=3.0)
    wd.beat(1_000_000)
    assert not wd.is_stale(1_000_000 + 2 * TF_MS)
    assert wd.is_stale(1_000_000 + 4 * TF_MS)


def test_watchdog_quiet_before_first_bar():
    wd = FeedWatchdog(timeframe_ms=TF_MS)
    assert not wd.is_stale(9_999_999_999)


# ---------- dashboard ----------

def test_dashboard_writes_selfcontained_html(tmp_path):
    writer = DashboardWriter(tmp_path / "dash.html", mode="replay")
    writer.write(
        equity_values=[10_000.0, 10_050.0, 9_990.0],
        initial_equity=10_000.0, closed_trades=2, win_rate=0.5,
        current_regime="bull", halted=False,
        positions=pd.DataFrame([{"symbol": "BTC/USDT", "direction": 1, "size": 0.1,
                                 "entry": 50_000.0, "stop": 49_800.0, "target": 50_300.0}]),
        weights=pd.DataFrame({"momentum": {"bull": 0.3}, "mean_reversion": {"bull": 0.7}}),
        session_table=pd.DataFrame([{"regime": "bull", "session": "us", "trades": 2,
                                     "win_rate": 0.5, "avg_r": 0.1, "total_pnl": 12.0}]),
        recent_trades=pd.DataFrame([{"symbol": "BTC/USDT", "pnl": 12.0}]),
    )
    html = (tmp_path / "dash.html").read_text()
    assert "<svg" in html and "polyline" in html
    assert "BTC/USDT" in html and "momentum" in html
    assert "bull" in html


def test_equity_svg_handles_short_series():
    assert "waiting" in equity_svg([10_000.0])
