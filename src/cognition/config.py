"""Configuration loading: secrets from environment/.env, everything else
from config/config.yaml. Two separate sources on purpose — secrets must
never live in a file that gets committed to git.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Secrets(BaseSettings):
    """Loaded from environment variables / .env. Never persisted to disk
    by this codebase, never logged.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    bybit_env: Literal["testnet", "live"] = "testnet"
    bybit_ip_whitelist: str = ""

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Explicit, single-purpose override required to run against live Bybit.
    # Absence of this blocks live trading at config-load time, independent
    # of anything an agent or strategy decides.
    allow_live_trading: str = ""

    @model_validator(mode="after")
    def _guard_live_mode(self) -> "Secrets":
        if self.bybit_env == "live" and self.allow_live_trading != "I_UNDERSTAND_THE_RISK":
            raise ValueError(
                "BYBIT_ENV=live requires ALLOW_LIVE_TRADING=I_UNDERSTAND_THE_RISK "
                "in the environment. This project must not place live orders "
                "until the product owner explicitly approves it."
            )
        return self


class ExchangeConfig(BaseModel):
    id: str = "bybit"
    category: str = "spot"


class HistoryConfig(BaseModel):
    years: int = 5
    fetch_limit: int = 1000
    request_pause_seconds: float = 0.2


class DataQualityConfig(BaseModel):
    max_gap_multiple: float = 2.0
    outlier_zscore: float = 6.0
    outlier_lookback: int = 200


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    log_dir: str = "logs"


class AppConfig(BaseModel):
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframes: list[str] = Field(default_factory=lambda: ["1m", "5m"])
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def resolve_path(self, relative: str) -> Path:
        path = PROJECT_ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def raw_dir(self) -> Path:
        return self.resolve_path(self.paths.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolve_path(self.paths.processed_dir)

    @property
    def log_dir(self) -> Path:
        return self.resolve_path(self.paths.log_dir)


def load_app_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)


@lru_cache
def get_settings() -> tuple[AppConfig, Secrets]:
    """Cached accessor for the merged config. Secrets are validated
    (including the live-trading guard) the first time this is called.
    """
    return load_app_config(), Secrets()
