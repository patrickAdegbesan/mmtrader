# Project status and handoff notes

Read this first. It records what has been established, what was ruled
out, and the traps already found — so none of it gets rediscovered the
expensive way.

## Where the project stands

All seven milestones are built and tested (177 passing). The system can
download real market history, backtest with realistic costs, train an RL
ensemble, enforce hard risk limits, paper trade off a live feed, and
retrain itself on a schedule with gated promotion.

**No money has ever been at risk, and no live order has ever been
placed.** Live trading is blocked twice over: at config load
(`BYBIT_ENV=live` requires `ALLOW_LIVE_TRADING=I_UNDERSTAND_THE_RISK`)
and again in the executor, which refuses a non-testnet client.

## The central finding — read before proposing strategy work

**Technical indicators on BTC price alone do not produce a tradeable
edge at the timeframes tested.** This is a measured result, not a
suspicion:

- Feature/label diagnostics showed roughly a 0.6% spread between
  best and worst feature quintiles against forward returns.
- Round-trip cost is 0.4%–1.2% depending on volatility regime
  (taker fees plus adverse slippage on both legs).
- The signal is smaller than the cost floor. No amount of model
  tuning fixes that; it is arithmetic.

Consequences worth internalising:

- A trained agent "learning to stop trading" on this data is the
  **correct** response to a market with no exploitable edge, not a
  training failure.
- Adding indicators, layers, or episodes will not change the result.
  The next idea has to change the *premise*, not the model.

The open candidate is **funding-rate / basis carry** — a delta-neutral
long-spot / short-perp position collecting the structural funding
premium. It sidesteps price prediction entirely. It has not been
analysed yet. The honest first step is analysis only: pull Bybit funding
history (public, no credentials), subtract fees on both legs, the
spot/perp basis, and rebalancing costs, then check what net carry
actually was across 2021–2026 including negative-funding stretches and
the Nov 2022 period. Do not build a strategy before that number exists.
It is not free money: it carries exchange risk, liquidation risk on the
short leg, and needs leverage/liquidation modelling the risk engine does
not currently have.

## Traps already found (do not re-introduce)

- **Testnet candles are synthetic.** A full year pulled from testnet had
  23.9% zero-volume bars, 44% stale bars (o==h==l==c), and a max
  "volume" of 4.3M BTC. Public market data must always come from
  mainnet; `BybitClient(..., public_data_only=True)` enforces this and
  refuses credentials so it cannot trade. `BYBIT_ENV` governs order
  routing only.
- **Paper trading needs real prices too.** It simulates its own fills
  and never sends an order, so a testnet feed would make an entire
  multi-week campaign fictional while appearing to work.
- **`is_clean` must compare fractions, not demand zeros.** Requiring
  zero outliers made it permanently false on real data (520 outliers in
  525k real candles is 0.1% — ordinary volatility). A flag that is
  always red is as useless as one always green.
- **Training config must match the data size.** `episode_bars` sized for
  a 20k-bar synthetic series covers ~2% of a five-year 1m split while
  finishing in minutes and reporting success.
- **Module loggers only reach a file after `configure_file_logging()`.**
  Grepping a log for a module's lines proves nothing unless a CLI
  configured file logging first.
- **One writer per data file.** `download()` holds its own in-memory
  copy and overwrites wholesale on save, so two concurrent runs mean the
  later one silently discards the other's work.

## Environment notes

Bybit reachability varies by environment, and `curl` succeeding is not
proof that Python can reach it — check with the actual client. Sandboxes
may allow testnet but not mainnet, or foreground but not background
tasks. Only a test from the machine that will do the work counts.

## Funding-carry analysis: started, not finished

A first probe exists — see [`research/FINDINGS.md`](../research/FINDINGS.md)
and `research/basis_probe.py`. Headline: across 8 sampled days 2023–2026 the
spot/perp basis averaged **−1.69bp** (only 3/8 days positive), i.e. the wrong
sign for long-spot/short-perp, at a magnitude an order below the ~31bp
round-trip fee.

Treat that as discouraging, not decided. It measures a trade-VWAP proxy rather
than realised funding, on 8 days out of ~1400. The analysis STATUS asks for is
still open.

Two things it did settle:

- `public.bybit.com` (tick archive, 2020→present for perp, 2022-11→present for
  spot) is **not** geo-blocked even where `api.bybit.com` and Binance are.
  That is the way in for historical work from a restricted environment.
- Minimum lot is 0.001 BTC, confirmed from tick data. Since the spot leg
  cannot be leveraged, the smallest constructible hedge is ~2× that notional
  (~$126 at $62.9k BTC).

## What to do next

1. Finish the funding-carry analysis — real funding history, full date range,
   fees and rebalancing (analysis only, no strategy code), **or**
2. Stop. The codebase is correct and the original question has a clean,
   well-evidenced answer. That is a legitimate place to end.

Anything that amounts to "try more indicators on candles" has already
been answered.
