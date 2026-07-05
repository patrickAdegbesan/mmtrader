"""Telegram alerting (Guardian B).

Uses the plain Bot API over HTTPS (urllib — honors HTTPS_PROXY) rather
than a heavyweight bot framework; we only ever send messages. If no
token/chat is configured, alerts degrade gracefully to the structured
log, so the trading loop never depends on Telegram availability.

Per-key throttling stops an alert storm (e.g. a flapping feed) from
flooding the chat: repeats of the same alert kind within the throttle
window are counted and summarized on the next send.
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Callable

from cognition.utils.logging import get_logger, log_with_fields

logger = get_logger("alerts")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _default_transport(token: str, chat_id: str, text: str) -> None:
    payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
    request = urllib.request.Request(
        TELEGRAM_API.format(token=token), data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


class TelegramAlerter:
    def __init__(
        self,
        token: str = "",
        chat_id: str = "",
        throttle_seconds: float = 300.0,
        transport: Callable[[str, str, str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.token = token
        self.chat_id = chat_id
        self.throttle_seconds = throttle_seconds
        self._transport = transport or _default_transport
        self._clock = clock
        self._last_sent: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}
        self.sent_count = 0

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def alert(self, kind: str, message: str, critical: bool = False) -> bool:
        """Returns True if the message went out (or would have, were
        Telegram configured); False if throttled."""
        now = self._clock()
        last = self._last_sent.get(kind)
        if not critical and last is not None and now - last < self.throttle_seconds:
            self._suppressed[kind] = self._suppressed.get(kind, 0) + 1
            return False

        suppressed = self._suppressed.pop(kind, 0)
        if suppressed:
            message += f" (+{suppressed} similar suppressed)"
        prefix = "🚨 CRITICAL: " if critical else "⚠️ "
        text = f"{prefix}[{kind}] {message}"

        log_with_fields(logger, 40 if critical else 30, "ALERT", kind=kind, detail=message, critical=critical)
        if self.configured:
            try:
                self._transport(self.token, self.chat_id, text)
            except Exception as exc:
                log_with_fields(logger, 40, "Telegram send failed", error=str(exc))
        self._last_sent[kind] = now
        self.sent_count += 1
        return True


class FeedWatchdog:
    """Detects a stalled market feed: if no bar arrives for
    `timeframe x stale_multiple` (wall-clock), something is wrong —
    exchange outage, dead WebSocket, or rate-limit lockout."""

    def __init__(self, timeframe_ms: int, stale_multiple: float = 3.0):
        self.timeout_ms = int(timeframe_ms * stale_multiple)
        self._last_bar_ms: int | None = None

    def beat(self, now_ms: int) -> None:
        self._last_bar_ms = now_ms

    def is_stale(self, now_ms: int) -> bool:
        if self._last_bar_ms is None:
            return False
        return now_ms - self._last_bar_ms > self.timeout_ms
