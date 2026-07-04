# Project Cognition

Self-learning, ensemble-based AI crypto scalping system for Bybit
(BTC/USDT, ETH/USDT). Built milestone by milestone per the architecture
document; this repo currently implements **Milestone 1 — Data Foundation**
only.

## Status: Milestone 1 — Data Foundation

Implements:
- Project scaffold, YAML config + `.env` secrets (pydantic-validated)
- Bybit connectivity via CCXT (testnet by default; live requires an
  explicit `ALLOW_LIVE_TRADING` env override — see below)
- Historical OHLCV downloader with **resume support** (re-running after
  an interruption only fetches new candles)
- Data-quality checks: duplicate timestamps, missing-candle gaps, outlier
  price moves
- Feature extraction: price velocity/acceleration, RSI, MACD, moving
  averages, Bollinger position, volume delta, volatility regime, session
  tag. (Order-book imbalance / spread width columns are present but NaN —
  they need live order-book data, which lands in a later milestone.)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: add Bybit TESTNET API key/secret (trade permission only,
# no withdrawal permission). Leave BYBIT_ENV=testnet.
```

Get testnet keys at https://testnet.bybit.com (Account → API Management).
Historical OHLCV downloads don't require keys at all (public endpoint) —
you only need them once execution/paper-trading milestones land.

## Run it (the one command)

```bash
python scripts/download_data.py
```

This downloads 5 years of 1m and 5m candles for BTC/USDT and ETH/USDT,
runs data-quality checks, extracts features, and writes:

- `data/raw/<SYMBOL>_<TIMEFRAME>.parquet` — clean raw OHLCV
- `data/processed/<SYMBOL>_<TIMEFRAME>.parquet` — feature-enriched dataset
- `logs/download_data.log` — structured JSON log of the whole run

Useful flags for a quick smoke test instead of a full 5-year pull:

```bash
python scripts/download_data.py --symbols BTC/USDT --timeframes 1m --years 1
```

Re-running the same command later only fetches candles newer than what's
already saved (resume, not re-download).

## Verify it worked

```bash
source .venv/bin/activate
python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/BTC_USDT_1m.parquet')
print(df.shape)
print(df.columns.tolist())
print(df.tail())
"
```

You should see OHLCV columns plus all the engineered feature columns
listed above, and `df.shape[0]` should roughly match 5 years of 1-minute
candles (minus any real exchange downtime).

Check `logs/download_data.log` for the quality report per symbol/timeframe
— it logs candle counts, duplicate counts, gap counts, and outlier counts.

## Tests

```bash
source .venv/bin/activate
python -m pytest -q
```

27 tests cover: config loading and the live-trading safety guard, the
Bybit client wrapper (sandbox mode, credential handling, retry behavior
on network errors, mocked — no real network calls), the downloader's
resume logic (mocked), data-quality detection (gaps/duplicates/outliers
on synthetic data), and the feature pipeline.

## Known limitation in this environment

This was built inside a network-sandboxed Claude Code session. The
sandbox's egress policy blocks direct calls to `api.bybit.com` and
`api-testnet.bybit.com` (confirmed via the proxy status endpoint — a
policy denial, not a bug in this code). That means **the live download
command above has not been executed end-to-end against the real Bybit
API from within this session** — only unit-tested against mocked
responses. Run `scripts/download_data.py` yourself on a machine/VPS/
environment with open network access to Bybit to do the real pull; the
code path is identical.

## Safety guardrails already in place

- `BYBIT_ENV=live` is rejected at config-load time unless
  `ALLOW_LIVE_TRADING=I_UNDERSTAND_THE_RISK` is also set in the
  environment — this project will not talk to live Bybit by accident.
- API keys are read only from environment variables / `.env` (gitignored),
  never hard-coded, never logged.

## Next milestone

Milestone 2 — Backtesting Engine (cost-realistic simulator, walk-forward
harness, Monte Carlo module, regime-split reporting) — **on hold pending
your review and approval of Milestone 1.**
