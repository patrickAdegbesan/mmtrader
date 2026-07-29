"""Historical OHLCV downloader with resume support.

Data is stored one parquet file per (symbol, timeframe) under
`data/raw/`. On each run we only fetch candles newer than what's already
on disk, so re-running the same command after an interruption (rate
limit, network drop, killed process) picks up where it left off instead
of re-downloading years of history.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from cognition.config import AppConfig
from cognition.data.bybit_client import BybitClient
from cognition.utils.logging import get_logger, log_with_fields

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

logger = get_logger("downloader")


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '_')}_{timeframe}.parquet"


class HistoricalDownloader:
    def __init__(self, client: BybitClient, config: AppConfig):
        self.client = client
        self.config = config

    def _file_path(self, symbol: str, timeframe: str) -> Path:
        return self.config.raw_dir / _symbol_to_filename(symbol, timeframe)

    def _load_existing(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._file_path(symbol, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _fetch_range(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        timeframe_ms: int,
        segment: str,
    ) -> list[list[float]]:
        """Page through candles in [start_ms, end_ms). Batches may overrun
        end_ms; the caller dedupes, so overlap is harmless.
        """
        rows: list[list[float]] = []
        cursor = start_ms
        limit = self.config.history.fetch_limit
        pause = self.config.history.request_pause_seconds

        while cursor < end_ms:
            batch = self.client.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            if not batch:
                break

            rows.extend(batch)
            last_ts = int(batch[-1][0])

            if last_ts < cursor:
                # Exchange returned nothing new; avoid spinning forever.
                break
            next_cursor = last_ts + timeframe_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(rows) % (limit * 10) == 0:
                log_with_fields(
                    logger, 20, "Download progress",
                    symbol=symbol, timeframe=timeframe, segment=segment, candles_fetched=len(rows),
                )
            time.sleep(pause)
        return rows

    def download(self, symbol: str, timeframe: str, years: int | None = None) -> pd.DataFrame:
        years = years or self.config.history.years
        timeframe_ms = self.client.parse_timeframe_ms(timeframe)
        now_ms = self.client.milliseconds()
        requested_since = now_ms - years * 365 * 24 * 60 * 60 * 1000

        existing = self._load_existing(symbol, timeframe)
        new_rows: list[list[float]] = []

        if existing is not None and len(existing) > 0:
            existing_min = int(existing["timestamp"].min())
            existing_max = int(existing["timestamp"].max())

            # Widening the window backwards must actually fetch the earlier
            # candles. Resuming forward-only would return a shorter history
            # than asked for and still log success — so asking for 5 years
            # on top of a 1-year file would silently leave you with 1.
            if requested_since < existing_min - timeframe_ms:
                log_with_fields(
                    logger, 20, "Backfilling earlier history",
                    symbol=symbol, timeframe=timeframe, years=years,
                    requested_since=requested_since, existing_earliest=existing_min,
                )
                new_rows.extend(self._fetch_range(
                    symbol, timeframe, requested_since, existing_min, timeframe_ms, "backfill",
                ))

            forward_since = existing_max + timeframe_ms
            if forward_since < now_ms:
                log_with_fields(
                    logger, 20, "Resuming download",
                    symbol=symbol, timeframe=timeframe,
                    resume_from=forward_since, existing_candles=len(existing),
                )
                new_rows.extend(self._fetch_range(
                    symbol, timeframe, forward_since, now_ms, timeframe_ms, "forward",
                ))
            elif not new_rows:
                log_with_fields(logger, 20, "Already up to date", symbol=symbol, timeframe=timeframe)
                return existing
        else:
            log_with_fields(
                logger, 20, "Starting fresh download",
                symbol=symbol, timeframe=timeframe, since=requested_since, years=years,
            )
            new_rows.extend(self._fetch_range(
                symbol, timeframe, requested_since, now_ms, timeframe_ms, "initial",
            ))

        new_df = pd.DataFrame(new_rows, columns=OHLCV_COLUMNS)
        combined = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
        combined = (
            combined.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        path = self._file_path(symbol, timeframe)
        combined.to_parquet(path, index=False)
        log_with_fields(
            logger, 20, "Saved candles",
            symbol=symbol, timeframe=timeframe, new_candles=len(new_df), total_candles=len(combined), path=str(path),
        )
        return combined

    def download_all(self) -> dict[tuple[str, str], pd.DataFrame]:
        results: dict[tuple[str, str], pd.DataFrame] = {}
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                results[(symbol, timeframe)] = self.download(symbol, timeframe)
        return results
