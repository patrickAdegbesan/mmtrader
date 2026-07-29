#!/usr/bin/env python3
"""One-command entrypoint for Milestone 1: download historical OHLCV,
run data-quality checks, extract features, and save enriched data.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --symbols BTC/USDT --timeframes 1m
    python scripts/download_data.py --years 1   # smaller pull for a quick smoke test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cognition.config import get_settings  # noqa: E402
from cognition.data.bybit_client import BybitClient  # noqa: E402
from cognition.data.downloader import HistoricalDownloader  # noqa: E402
from cognition.data.quality import clean_ohlcv, run_quality_checks  # noqa: E402
from cognition.features.engineering import extract_features  # noqa: E402
from cognition.utils.logging import configure_file_logging, get_logger, log_with_fields  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated, e.g. BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default=None, help="Comma-separated, e.g. 1m,5m")
    parser.add_argument("--years", type=int, default=None, help="Override years of history to fetch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, secrets = get_settings()

    if args.symbols:
        config.symbols = [s.strip() for s in args.symbols.split(",")]
    if args.timeframes:
        config.timeframes = [t.strip() for t in args.timeframes.split(",")]

    configure_file_logging(config.log_dir, "download_data")
    logger = get_logger("download_data")
    log_with_fields(
        logger, 20, "Milestone 1 pipeline starting",
        symbols=config.symbols, timeframes=config.timeframes, testnet=secrets.bybit_env == "testnet",
    )

    # Historical candles are public mainnet data. Never testnet: those
    # candles are synthetic and would poison every downstream model.
    client = BybitClient(config, secrets, public_data_only=True)
    downloader = HistoricalDownloader(client, config)

    exit_code = 0
    for symbol in config.symbols:
        for timeframe in config.timeframes:
            raw_df = downloader.download(symbol, timeframe, years=args.years)
            if raw_df.empty:
                log_with_fields(logger, 30, "No data downloaded", symbol=symbol, timeframe=timeframe)
                continue

            cleaned_df = clean_ohlcv(raw_df)
            timeframe_ms = client.parse_timeframe_ms(timeframe)
            report = run_quality_checks(cleaned_df, symbol, timeframe, timeframe_ms, config.data_quality)
            log_with_fields(
                logger, 20 if report.is_clean else 30, "Quality report",
                symbol=symbol, timeframe=timeframe, summary=report.summary(), is_clean=report.is_clean,
            )
            if not report.is_clean:
                exit_code = exit_code or 0  # quality issues are logged, not fatal — see README

            featured_df = extract_features(cleaned_df)
            out_path = config.processed_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"
            featured_df.to_parquet(out_path, index=False)
            log_with_fields(
                logger, 20, "Saved processed dataset",
                symbol=symbol, timeframe=timeframe, rows=len(featured_df), columns=len(featured_df.columns), path=str(out_path),
            )

    log_with_fields(logger, 20, "Milestone 1 pipeline complete")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
