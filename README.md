# Project Cognition

Self-learning, ensemble-based AI crypto scalping system for Bybit
(BTC/USDT, ETH/USDT). Built milestone by milestone per the architecture
document; this repo currently implements **Milestones 1–2**.

## Status: Milestone 5 — Risk Engine + Execution

Implements Guardian A and the order-execution module:

- **Risk Engine** (`src/cognition/risk/engine.py`): standalone gate —
  every proposed trade passes `evaluate()` BEFORE execution, agents can
  never override it. Enforces all non-negotiables:
  1. ≤1% equity risk per trade, fractional-Kelly sized within the cap
     (no-edge proposals are rejected outright);
  2. daily circuit breaker at −5%: halts everything for the rest of the
     UTC day (review mode), resets on day rollover;
  3. drawdown governor: sizes shrink automatically while drawdown from
     peak exceeds 10%;
  4. overtrading guard: sliding max-trades-per-hour/day windows;
  5. exposure limits: max concurrent positions, no doubling into a
     symbol, total-notional cap (oversized orders are shrunk to fit);
  6. model-proposed SL/TP clamped to hard bounds; proposals without a
     stop are rejected.
  Every decision carries an audit trail (reason + adjustments), and the
  **config validators refuse values looser than the non-negotiables** —
  `risk_per_trade: 0.02` in YAML fails at load time.
- **Limit-first executor** (`src/cognition/execution/executor.py`):
  quotes as a maker (post-only limit at the near touch), polls for the
  fill, cancels on timeout and sends a market order only for the
  unfilled remainder — with cancel/fill race reconciliation so a fill
  that lands mid-cancel is never doubled. Order placement is never
  blind-retried (a timed-out create may still have reached the exchange).
- **Testnet-only, twice**: the executor refuses a non-testnet client
  unless the explicit `ALLOW_LIVE_TRADING` override is set — independent
  of the same guard at config load.
- **Bybit order API** on the client wrapper: create/cancel/fetch order,
  fetch ticker, with retry policies (reads retry; order placement does not).

Run it:

```bash
python scripts/testnet_trade_demo.py            # dry-run: full pipeline on a mock exchange
python scripts/testnet_trade_demo.py --testnet  # real TESTNET order (needs keys + network)
```

The demo walks a sane proposal, a reckless stop (clamped), a no-edge
proposal (rejected), and the circuit breaker tripping + holding through
recovery — then executes the approved trade limit-first with market
fallback.

## Milestone 4 — Full Ensemble + Meta-Learner (complete)

Implements the 5-agent ensemble:

- **Four market agents** (`src/cognition/agents/specs.py`), each an
  independent DQN seeing only its own feature lens: momentum
  (velocity/acceleration/MACD), mean-reversion (RSI/Bollinger/SMA
  distance), volume/breakout (volume z-score/Donchian position), and
  microstructure. Microstructure currently trains on candle-derived
  PROXIES (range width ~ spread, close-in-range ~ imbalance) because
  historical OHLCV has no order-book data; real features replace them
  when the live WebSocket feed lands (Milestone 6).
- **Meta-learner with regime memory** (`meta.py`): watches rolling
  per-agent performance and reweights votes — kept PER MARKET REGIME, so
  momentum can dominate trends while mean-reversion owns ranges, learned
  not hard-coded. Online update after every closed trade:
  agents that agreed with winners gain, agents that backed losers lose,
  scaled by their confidence; scores decay so recent performance rules.
  Deliberately transparent (EWMA + softmax) so every weight is auditable.
- **Voting** (`ensemble.py`): per bar, each agent votes direction +
  confidence (softmax of its Q-values) + its preferred vol-adjusted
  SL/TP. Combined score = Σ weight × confidence × direction; trades fire
  above a threshold; SL/TP is the confidence-weighted average of the
  agreeing agents' levels. Every decision (votes, weights, score) goes
  to `decision_log` — the Guardian-B audit trail.
- **Per-(regime, session) performance tracking** (`SessionPerformanceTracker`):
  the "works in European hours, fails in Asian hours" memory.
- **Engine close-hook**: the backtester now notifies strategies via
  `on_trade_closed()` after every close — that's what drives the
  meta-learner's online learning inside any backtest or (later) live run.

Run it:

```bash
python scripts/train_ensemble.py                    # synthetic demo
python scripts/train_ensemble.py --data data/processed/BTC_USDT_1m.parquet
```

Trains all four agents (versioned checkpoints per agent under
`models/<agent>/`), assembles the ensemble, runs it through the
cost-inclusive backtester with the meta-learner updating online, then
prints: per-agent learning improvement, ensemble metrics, meta weights
per regime, the (regime, session) table, and a sample audit-log entry.

## Milestone 3 — Single Agent Prototype (complete)

Implements the momentum agent trained end-to-end with the learning loop
verified against the backtester:

- **RL environment** (`src/cognition/agents/env.py`): gym-style, steps
  bar-by-bar with the SAME cost arithmetic as the backtester (next-bar
  execution, adverse regime-aware slippage, fees, stop-before-target).
  Reward = R-multiple of each closed trade after all costs (risk-adjusted,
  per the spec — never raw profit), plus a small holding penalty.
- **DQN with actor-critic components** (`networks.py`, `dqn.py`): dueling
  architecture — shared trunk feeding a state-value head (critic) and an
  advantage-over-actions head (actor stream); Double-DQN updates, replay
  buffer, target network, epsilon-greedy exploration.
- **Learned, volatility-adjusted SL/TP** (`actions.py`): actions carry
  stop tightness; stops scale with current volatility, clamped to
  configured bounds (the Risk Engine adds hard clamps in Milestone 5).
  Sizing always risks ≤1% of equity — enforced in the env exactly as in
  the backtester.
- **Versioned model registry** (`src/cognition/learning/registry.py`):
  every checkpoint is an immutable version with metadata; LATEST is a
  movable pointer, so rollback = activate an older version. No silent
  updates possible.
- **Trainer** (`trainer.py`): episodes over random train-slice windows,
  periodic greedy evaluation on a held-out tail slice (normalization
  stats fitted on train only — no leakage), checkpoint per eval.
- **Backtester adapter** (`momentum.py`): a trained agent plugs into the
  Milestone-2 engine as a Strategy, so walk-forward/Monte Carlo/regime
  reports all work on agents.

Run it:

```bash
python scripts/train_momentum.py                    # synthetic demo
python scripts/train_momentum.py --data data/processed/BTC_USDT_1m.parquet
```

Prints the learning curve (untrained baseline → periodic evals), saves
versioned checkpoints under `models/momentum/`, then runs the trained
agent through the cost-inclusive backtester. Exit code 0 iff the final
eval beats the untrained baseline.

The pytest acceptance test (`tests/test_learning_loop.py`) trains a
seeded agent on a strongly trending synthetic market and asserts the
learning loop (a) improves eval reward over the untrained baseline and
(b) reaches profitability after costs on that easy market.

## Milestone 2 — Backtesting Engine (complete)

Implements:
- **Cost-realistic simulator** (`src/cognition/backtest/simulator.py`):
  event-driven, per-bar engine. Taker fees (0.1%), regime-aware slippage
  (0.1% base / 0.5% in high-volatility candles), order latency (signals
  fill at the *next* bar open — no lookahead), partial fills, conservative
  intra-bar ordering (stop assumed hit before target), gap handling.
  Position sizing risks a fixed ≤1% of equity per trade off the stop
  distance; the 1% cap is enforced in code and cannot be configured higher.
- **Walk-forward harness**: train window → test window → roll forward;
  test windows get warmup bars for indicators but can't trade in them.
- **Out-of-sample lockbox**: most recent 6 months split off before
  anything touches the data.
- **Monte Carlo robustness**: volatility-scaled price-path perturbation,
  N runs, percentile summary.
- **Regime-split reporting**: trades bucketed by bull/bear/sideways
  (trailing-return labels, no lookahead) and by session, with the
  architecture doc's gate-to-live pass criteria checked explicitly.
- **Dummy SMA-crossover strategy** to prove the plumbing (it loses money
  after costs — expected; that's the cost model working, not a bug).

Run it:

```bash
python scripts/run_backtest.py                      # synthetic-data demo
python scripts/run_backtest.py --data data/processed/BTC_USDT_1m.parquet
```

Prints the full report and saves it under `reports/`. A custom
event-driven engine was chosen over VectorBT/Backtrader because the RL
agents (Milestone 3) need a stepping environment to train against — one
engine, one cost model, shared by backtesting and training.

## Milestone 1 — Data Foundation (complete)

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

136 tests cover: config loading and the live-trading safety guard, the
Bybit client wrapper (sandbox mode, credential handling, retry behavior
on network errors, mocked — no real network calls), the downloader's
resume logic (mocked), data-quality detection (gaps/duplicates/outliers
on synthetic data), the feature pipeline, and the backtest framework —
cost model arithmetic (fees, adverse slippage both directions, regime
widening), engine mechanics (latency fills, stop/target/both-hit/gap
handling, risk-capped sizing, partial fills, force close, no-lookahead),
metrics, regime labeling, walk-forward window integrity, lockbox split,
and Monte Carlo reproducibility. The RL stack adds: action/SL-TP mapping
and clamps, env execution honesty (next-bar fills, slippage, 1% risk cap
enforced), reward = post-cost R-multiple, replay/network/agent mechanics,
save/load determinism, registry versioning + rollback, and the seeded
learning-improvement acceptance test.

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

Milestone 6 — Paper Trading Mode (full live-data paper trading on Bybit
Testnet with monitoring, alerts, and the learning loop in real time) —
**on hold pending review and approval of Milestone 5.**
