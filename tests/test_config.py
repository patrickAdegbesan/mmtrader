import pytest
from pydantic import ValidationError

from cognition.config import AppConfig, Secrets, load_app_config


def test_app_config_defaults():
    config = AppConfig()
    assert config.symbols == ["BTC/USDT", "ETH/USDT"]
    assert config.timeframes == ["1m", "5m"]
    assert config.history.years == 5


def test_load_app_config_from_yaml_file():
    config = load_app_config()
    assert "BTC/USDT" in config.symbols
    assert config.exchange.id == "bybit"


def test_resolve_path_creates_directory():
    path = AppConfig().resolve_path("data/raw")
    assert path.exists()
    assert path.is_dir()


def test_secrets_default_is_testnet(monkeypatch):
    monkeypatch.delenv("BYBIT_ENV", raising=False)
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    secrets = Secrets(_env_file=None)
    assert secrets.bybit_env == "testnet"


def test_secrets_live_mode_blocked_without_explicit_override(monkeypatch):
    monkeypatch.setenv("BYBIT_ENV", "live")
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    with pytest.raises(ValidationError):
        Secrets(_env_file=None)


def test_secrets_live_mode_allowed_with_explicit_override(monkeypatch):
    monkeypatch.setenv("BYBIT_ENV", "live")
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "I_UNDERSTAND_THE_RISK")
    secrets = Secrets(_env_file=None)
    assert secrets.bybit_env == "live"
