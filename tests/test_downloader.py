from unittest.mock import MagicMock

import pandas as pd
import pytest

from cognition.config import AppConfig
from cognition.data.downloader import HistoricalDownloader

TIMEFRAME_MS = 60_000
YEAR_MS = 365 * 24 * 60 * 60 * 1000


def make_candles(start_ts, count):
    return [[start_ts + i * TIMEFRAME_MS, 100.0, 101.0, 99.0, 100.5, 10.0] for i in range(count)]


@pytest.fixture
def config(tmp_path):
    cfg = AppConfig()
    cfg.paths.raw_dir = str(tmp_path / "raw")
    cfg.history.request_pause_seconds = 0
    cfg.history.fetch_limit = 50
    return cfg


def make_mock_client(pages, now_ms):
    client = MagicMock()
    client.parse_timeframe_ms.return_value = TIMEFRAME_MS
    client.milliseconds.return_value = now_ms
    client.fetch_ohlcv.side_effect = pages
    return client


def test_fresh_download_paginates_until_caught_up(config):
    now_ms = 1_700_000_000_000 + 150 * TIMEFRAME_MS
    page1 = make_candles(1_700_000_000_000, 50)
    page2 = make_candles(page1[-1][0] + TIMEFRAME_MS, 50)
    page3 = make_candles(page2[-1][0] + TIMEFRAME_MS, 50)
    client = make_mock_client([page1, page2, page3, []], now_ms)

    downloader = HistoricalDownloader(client, config)
    df = downloader.download("BTC/USDT", "1m", years=1)

    assert len(df) == 150
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].duplicated().sum() == 0


def test_empty_page_mid_history_is_treated_as_a_gap_not_the_end(config):
    """Confirmed on real Bybit data: BTC/USDT 5m klines have zero candles
    for ~4 days around 2022-04-11 while the 1m series for the same window
    is complete. An empty page must not stop the whole download — it
    silently truncated years of later history the first time this hit.
    """
    limit = config.history.fetch_limit  # 50
    page1 = make_candles(1_700_000_000_000, 50)

    gap_query_since = page1[-1][0] + TIMEFRAME_MS
    skip_to = gap_query_since + limit * TIMEFRAME_MS
    page2 = make_candles(skip_to, 50)
    now_ms = page2[-1][0] + TIMEFRAME_MS  # loop exits right after page2, no trailing call needed

    client = make_mock_client([page1, [], page2], now_ms)
    downloader = HistoricalDownloader(client, config)
    df = downloader.download("BTC/USDT", "1m", years=1)

    assert len(df) == 100
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].duplicated().sum() == 0
    # The gap itself must survive in the data — find_gaps() is what should
    # surface it later, not the downloader fabricating continuity.
    assert int(df["timestamp"].iloc[50]) - int(df["timestamp"].iloc[49]) > TIMEFRAME_MS


def seed_existing(downloader, rows):
    path = downloader._file_path("BTC/USDT", "1m")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_parquet(path)


def test_download_resumes_from_last_saved_timestamp(config):
    now_ms = 1_700_000_000_000
    # Existing data already reaches further back than the requested year,
    # so only the forward segment has anything to fetch.
    existing = make_candles(now_ms - YEAR_MS - 100 * TIMEFRAME_MS, 50)
    resume_start = existing[-1][0] + TIMEFRAME_MS
    # Sized so this single batch's next cursor lands exactly on now_ms —
    # an empty page no longer ends a fetch loop (see the gap test above),
    # so the loop must end via reaching end_ms, not via a trailing [].
    gap_candles = (now_ms - resume_start) // TIMEFRAME_MS
    second_batch = make_candles(resume_start, gap_candles)

    client = make_mock_client([second_batch], now_ms)
    downloader = HistoricalDownloader(client, config)
    seed_existing(downloader, existing)
    result = downloader.download("BTC/USDT", "1m", years=1)

    # First call to fetch_ohlcv on resume must start after the saved data.
    assert client.fetch_ohlcv.call_args_list[0].kwargs["since"] == resume_start
    assert len(result) == 50 + gap_candles
    assert result["timestamp"].duplicated().sum() == 0


def test_widening_years_backfills_earlier_history(config):
    """Asking for more history than is on disk must fetch the earlier
    candles, not silently resume forward and report success.
    """
    now_ms = 1_700_000_000_000
    existing = make_candles(now_ms - 50 * TIMEFRAME_MS, 50)
    existing_min = existing[0][0]
    requested_since = now_ms - YEAR_MS
    # Sized so this single batch's next cursor lands exactly on
    # existing_min — an empty page no longer ends a fetch loop, so the
    # loop must end via reaching end_ms, not via a trailing [].
    gap_candles = (existing_min - requested_since) // TIMEFRAME_MS
    backfill = make_candles(requested_since, gap_candles)

    client = make_mock_client([backfill], now_ms)
    downloader = HistoricalDownloader(client, config)
    seed_existing(downloader, existing)
    result = downloader.download("BTC/USDT", "1m", years=1)

    # Fetching must begin at the requested start, not at the saved tail.
    assert client.fetch_ohlcv.call_args_list[0].kwargs["since"] == requested_since
    assert int(result["timestamp"].min()) == requested_since
    assert len(result) == 50 + gap_candles
    assert result["timestamp"].duplicated().sum() == 0


def test_already_up_to_date_skips_fetch(config):
    now_ms = 1_700_000_000_000
    # Earliest reaches past the requested year and latest is current, so
    # neither the backfill nor the forward segment has work to do.
    existing = [
        [now_ms - YEAR_MS - TIMEFRAME_MS, 100.0, 101.0, 99.0, 100.5, 10.0],
        [now_ms - TIMEFRAME_MS, 100.0, 101.0, 99.0, 100.5, 10.0],
    ]
    client = make_mock_client([], now_ms)
    downloader = HistoricalDownloader(client, config)
    seed_existing(downloader, existing)

    result = downloader.download("BTC/USDT", "1m", years=1)
    client.fetch_ohlcv.assert_not_called()
    assert len(result) == 2
