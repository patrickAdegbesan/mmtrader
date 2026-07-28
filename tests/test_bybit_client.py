from unittest.mock import MagicMock, patch

import ccxt
import pytest

from cognition.config import AppConfig, Secrets
from cognition.data.bybit_client import BybitClient


def make_client(bybit_env="testnet", api_key=""):
    config = AppConfig()
    secrets = Secrets(_env_file=None, bybit_env=bybit_env, bybit_api_key=api_key)
    return BybitClient(config, secrets)


def test_testnet_client_enables_sandbox_mode():
    client = make_client()
    exchange = client.exchange
    assert client.is_testnet is True
    # ccxt's sandbox mode swaps in testnet URLs
    assert "testnet" in str(exchange.urls.get("api", "")).lower() or exchange.urls.get("test")


def test_unauthenticated_client_has_no_credentials():
    client = make_client(api_key="")
    exchange = client.exchange
    assert not exchange.apiKey


def test_authenticated_client_carries_credentials():
    client = make_client(api_key="dummy-key")
    exchange = client.exchange
    assert exchange.apiKey == "dummy-key"


def test_public_data_client_uses_mainnet_even_when_env_is_testnet():
    """Historical candles and live klines must come from mainnet: testnet
    candles are synthetic, so training or paper trading on them measures
    fiction. This must hold even though BYBIT_ENV=testnet."""
    config = AppConfig()
    secrets = Secrets(_env_file=None, bybit_env="testnet")
    client = BybitClient(config, secrets, public_data_only=True)
    assert client.is_testnet is False
    assert "testnet" not in str(client.exchange.urls["api"]).lower()


def test_public_data_client_refuses_credentials_so_it_cannot_trade():
    config = AppConfig()
    secrets = Secrets(_env_file=None, bybit_env="testnet", bybit_api_key="k", bybit_api_secret="s")
    client = BybitClient(config, secrets, public_data_only=True)
    assert not client.exchange.apiKey
    assert not client.exchange.secret


def test_default_client_still_honours_testnet_for_order_paths():
    client = make_client(bybit_env="testnet")
    assert client.is_testnet is True
    assert "testnet" in str(client.exchange.urls["api"]).lower()


def test_fetch_ohlcv_retries_on_network_error_then_succeeds():
    client = make_client()
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.side_effect = [ccxt.NetworkError("timeout"), [[1, 2, 3, 4, 5, 6]]]
    client._exchange = mock_exchange

    result = client.fetch_ohlcv("BTC/USDT", "1m", since=0, limit=10)

    assert result == [[1, 2, 3, 4, 5, 6]]
    assert mock_exchange.fetch_ohlcv.call_count == 2


def test_fetch_ohlcv_does_not_retry_on_bad_symbol():
    client = make_client()
    mock_exchange = MagicMock()
    mock_exchange.fetch_ohlcv.side_effect = ccxt.BadSymbol("no such symbol")
    client._exchange = mock_exchange

    with pytest.raises(ccxt.BadSymbol):
        client.fetch_ohlcv("NOT/REAL", "1m")

    assert mock_exchange.fetch_ohlcv.call_count == 1
