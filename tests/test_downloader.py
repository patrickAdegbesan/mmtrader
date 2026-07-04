from unittest.mock import MagicMock

import pandas as pd
import pytest

from cognition.config import AppConfig
from cognition.data.downloader import HistoricalDownloader

TIMEFRAME_MS = 60_000


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


def test_download_resumes_from_last_saved_timestamp(config):
    now_ms = 1_700_000_000_000 + 100 * TIMEFRAME_MS
    first_batch = make_candles(1_700_000_000_000, 50)
    client = make_mock_client([first_batch, []], now_ms)
    downloader = HistoricalDownloader(client, config)
    first_df = downloader.download("BTC/USDT", "1m", years=1)
    assert len(first_df) == 50

    resume_start = first_batch[-1][0] + TIMEFRAME_MS
    second_batch = make_candles(resume_start, 50)
    client2 = make_mock_client([second_batch, []], now_ms)
    downloader2 = HistoricalDownloader(client2, config)
    second_df = downloader2.download("BTC/USDT", "1m", years=1)

    # First call to fetch_ohlcv on resume must start after the saved data.
    resume_call_kwargs = client2.fetch_ohlcv.call_args_list[0].kwargs
    assert resume_call_kwargs["since"] == resume_start
    assert len(second_df) == 100
    assert second_df["timestamp"].duplicated().sum() == 0


def test_already_up_to_date_skips_fetch(config):
    now_ms = 1_700_000_000_000
    existing = make_candles(now_ms - 10 * TIMEFRAME_MS, 10)
    client = make_mock_client([], now_ms)
    downloader = HistoricalDownloader(client, config)
    downloader._file_path("BTC/USDT", "1m").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(existing, columns=["timestamp", "open", "high", "low", "close", "volume"]).to_parquet(
        downloader._file_path("BTC/USDT", "1m")
    )

    result = downloader.download("BTC/USDT", "1m", years=1)
    client.fetch_ohlcv.assert_not_called()
    assert len(result) == 10
