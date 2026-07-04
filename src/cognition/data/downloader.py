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

    def download(self, symbol: str, timeframe: str, years: int | None = None) -> pd.DataFrame:
        years = years or self.config.history.years
        timeframe_ms = self.client.parse_timeframe_ms(timeframe)
        now_ms = self.client.milliseconds()

        existing = self._load_existing(symbol, timeframe)
        if existing is not None and len(existing) > 0:
            since_ms = int(existing["timestamp"].max()) + timeframe_ms
            log_with_fields(
                logger, 20, "Resuming download",
                symbol=symbol, timeframe=timeframe, resume_from=since_ms, existing_candles=len(existing),
            )
        else:
            since_ms = now_ms - years * 365 * 24 * 60 * 60 * 1000
            log_with_fields(
                logger, 20, "Starting fresh download",
                symbol=symbol, timeframe=timeframe, since=since_ms, years=years,
            )

        if since_ms >= now_ms:
            log_with_fields(logger, 20, "Already up to date", symbol=symbol, timeframe=timeframe)
            return existing if existing is not None else pd.DataFrame(columns=OHLCV_COLUMNS)

        new_rows: list[list[float]] = []
        cursor = since_ms
        limit = self.config.history.fetch_limit
        pause = self.config.history.request_pause_seconds

        while cursor < now_ms:
            batch = self.client.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            if not batch:
                break

            new_rows.extend(batch)
            last_ts = int(batch[-1][0])

            if last_ts < cursor:
                # Exchange returned nothing new; avoid spinning forever.
                break
            next_cursor = last_ts + timeframe_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(new_rows) % (limit * 10) == 0:
                log_with_fields(
                    logger, 20, "Download progress",
                    symbol=symbol, timeframe=timeframe, candles_fetched=len(new_rows),
                )
            time.sleep(pause)

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
