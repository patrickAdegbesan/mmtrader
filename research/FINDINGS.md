# Findings — BTC spot/perp carry on Bybit

Status: **smoke test only. Not a backtest. No capital should move on this.**

Prerequisite reading: [`docs/STATUS.md`](../docs/STATUS.md). This file is the
opening of the analysis pass that STATUS calls for under "What to do next" —
funding-rate carry, analysis only, no strategy code. STATUS's instruction
holds: *do not build a strategy before that number exists.* The number does
not exist yet. What follows is a first probe, not the number.

## The question

Long BTC spot + short BTCUSDT perp under Portfolio Margin, collecting funding.
Earlier working assumption was a ~3–5%/yr net carry premium. Does the premium
actually exist?

This is the successor question to the one STATUS already answered negatively:
technical indicators on BTC candles carry ~0.6% quintile spread against a
0.4–1.2% round-trip cost floor, so there is no edge there. Carry sidesteps
price prediction entirely, which is why it is the open candidate.

## Account setup (verified, done)

Portfolio Margin and Spot Hedging are both enabled on the live account. There
was no application, no approval queue, and no minimum-equity gate — it is a
self-serve toggle at **Assets → Unified Trading → the margin-mode pill**, not
in Account Settings. Prerequisites (UTA, no options positions, no borrowings,
no hedge-mode positions) were all trivially met on an empty account.

## Hard constraint (verified from tick data)

The smallest trades in the BTCUSDT perp archive are exactly 0.001 BTC, which
confirms the minimum lot. With BTC at $62,858 (2026-08-01 VWAP, from the data):

| | |
|---|---|
| min perp order | 0.001 BTC = **$62.86** notional |
| matching spot leg | **$62.86** — cannot be leveraged into existence |
| **minimum viable hedge** | **≈ $126** |

Spot is the binding leg. Perp margin can be levered down; spot cannot. Any
account that cannot fund ~$63 of unleveraged spot cannot construct the hedge
at all — it can only hold a naked leveraged perp, which is a different trade
with a different risk profile.

## First look at the basis

Eight days sampled across 2023–2026 via `basis_probe.py`:

```
date           spotVWAP   perpVWAP  basis_bp     ann%  hrs
2023-01-15      20776.0    20781.5      2.63    28.76   24
2023-07-15      30309.2    30293.2     -5.30   -58.01   24
2024-01-15      42615.4    42624.0      2.02    22.17   24
2024-07-15      62919.0    62904.2     -2.34   -25.61   24
2025-01-15      98100.4    98082.5     -1.83   -20.02   24
2025-07-15     117285.5   117291.8      0.54     5.92   24
2026-01-15      96290.0    96236.9     -5.52   -60.46   24
2026-07-15      64895.0    64871.0     -3.69   -40.45   24

mean basis -1.69 bp  ->  ~-18.46% APR gross
days positive: 3/8
```

Two observations:

1. **Sign is wrong on average.** Mean basis is negative — the perp traded
   *below* spot. For long-spot/short-perp, that means paying funding rather
   than collecting it. Only 3 of 8 days were positive.
2. **Magnitude is dwarfed by costs.** Daily basis is ±2–6 bp. Round-trip fees
   on the two-leg hedge are ~31 bp all-taker (spot 0.1 + perp 0.055, twice).

This does not support the 3–5%/yr premium assumption. It points the other way.

## Why this is not yet conclusive

Three real limitations, in rough order of severity:

1. **VWAP basis is a proxy for funding, not funding itself.** Bybit derives
   funding from the time-weighted impact bid/ask against the index price. Trade
   VWAP is biased by which side is aggressing. The realised cash flow could
   differ in sign from what is measured here. **This is the important one** —
   it means the headline result above may simply be measuring the wrong thing.
2. **Eight days out of ~1,400 is not a sample.** Daily values range +2.6 to
   −5.5 bp. The variance swamps the mean at this n.
3. **The `ann%` column overstates.** Annualising one day's 5 bp into −60% is
   arithmetically fine and rhetorically misleading. Read `basis_bp`.

## What a real backtest needs

STATUS scopes this as: pull funding history, subtract fees on both legs, the
spot/perp basis, and rebalancing costs, then check net carry across 2021–2026
*including negative-funding stretches and the Nov 2022 period*. Concretely:

- [ ] **Actual funding-rate history**, not the VWAP proxy. Available from
      `GET /v5/market/funding/history` on api.bybit.com. That host is
      geo-blocked from this container but reachable from Nigeria — must be
      pulled locally. Note STATUS's warning that reachability varies by
      environment and `curl` succeeding does not prove the Python client can
      reach it; verify from the machine that will do the work.
- [ ] Full date range rather than sampled days (spot∩perp = 2022-11 → present).
- [ ] Explicit fee model separating maker and taker on each leg.
- [ ] Slippage / fill model for the two-leg entry, including leg-in risk.
- [ ] Margin and liquidation simulation under Portfolio Margin offsets. STATUS
      flags that the risk engine has no leverage/liquidation modelling today,
      so this is new work rather than configuration.

Only after that does a minimum-capital number mean anything. Any figure quoted
before then — including the "$2,000" floated earlier in discussion — is a guess.

The $126 minimum-hedge figure above is different in kind: it is a mechanical
floor from lot size and spot's un-leverageable nature, and holds regardless of
whether the carry turns out to be positive.

## Environment note

`api.bybit.com` and `fapi.binance.com` both geo-block this container
(CloudFront / region restriction). `public.bybit.com` does **not**, which is
what made the work above possible. Anything live must run from the user's own
machine or a VPS in a served region.

---

# Open issue — the cost model contradicts the executor

Raised by the user, and it looks correct. Recording it because it may
overturn the central "no edge" finding in `docs/STATUS.md`.

## The contradiction

`src/cognition/execution/executor.py` posts **maker** orders by design. Its
own docstring:

> quote as a maker first (limit at the near touch, post-only where
> supported), poll for the fill, and only cross the spread with a market
> order for the unfilled remainder after a timeout — *scalping lives and
> dies on fees, so we pay taker only when we must.*

`src/cognition/backtest/costs.py` has no maker path at all:

```python
def round_trip_cost(self, volatility_regime):
    return 2 * self.taker_fee + 2 * self.slippage_rate(volatility_regime)
```

Taker on every fill, adverse slippage on both legs. **The backtest prices a
strategy the system does not run.**

## Why it may matter

| | round-trip cost |
|---|---|
| backtest as configured | 0.4% – 1.2% |
| perp maker, filled at the touch | ~0.04% |
| measured feature quintile spread | **0.6%** |

Against 0.4% the signal loses, and an agent that declines to trade is
behaving correctly — that is STATUS's conclusion. Against 0.04% the same
signal clears the floor by an order of magnitude. The conclusion may be an
artifact of the cost assumption rather than a property of the market.

## Why it may not

Three reasons not to get excited yet:

1. **Adverse selection is the real maker cost, and neither model has it.**
   Resting orders fill when the market is about to run you over and miss
   when it is about to go your way. A maker backtest that assumes fills at
   the touch is optimistic in exactly the way the current one is
   pessimistic. Fill probability has to be modelled before either number
   means anything.
2. **`config.yaml` sets `category: spot`.** Bybit spot charges maker and
   taker the same 0.1%, so on spot the maker path saves slippage but no
   fees. The fee saving above is a *perp* number — realising it is a change
   of venue, not just of order type.
3. **The executor falls back to taker** on the unfilled remainder after
   `limit_timeout_seconds`, so realised cost is a blend, not the maker rate.

## What would settle it

- [ ] Add maker/taker as separate rates in `CostModel` rather than one
      `taker_fee`, so a backtest can price the strategy the executor runs.
- [ ] Model fill probability for resting orders, including the case where
      the fill is itself the bad news. Without this, maker backtests lie.
- [ ] Re-run the feature/label diagnostic under the maker cost floor and see
      whether the 0.6% spread survives.
- [ ] Decide spot vs perp deliberately — the fee structures differ enough to
      change which strategies are viable.

Until then STATUS's "no edge" should be read as **"no edge at taker cost"**,
which is a narrower claim than it currently appears to make.

---

# Result — the maker floor does not rescue the signal

Ran the corrected diagnostic on 64,800 real Bybit 1m bars (2026-06-15 →
2026-07-29, tick archive aggregated by `research/fetch_bars.py`).

**The answer is no.** The cost model was genuinely wrong, fixing it changed
the floor by ~8x, and the signal still does not clear it.

## At scalping horizons

| horizon | best excess over null | maker floor | clears? |
|---|---|---|---|
| 5m | 0.0038% | 0.04% | 0/15 |
| 15m | 0.0052% | 0.04% | 0/15 |
| 60m | 0.0154% | 0.04% | 0/15 |

The best feature's excess is **~8x below** even the cheapest floor. For
scale, the same measurement on a synthetic random walk gave a best excess of
0.0340% — **real BTC showed less apparent signal than noise did.**

## At longer horizons, and why it is not a reprieve

| horizon | best excess | median null | clears maker |
|---|---|---|---|
| 240m | 0.0422% (volatility) | 0.0739% | 1/15 |
| 720m | 0.0649% (volatility) | 0.0861% | 2/15 |

This looks like something appears at 4–12h. It does not:

1. **The winner sits below the noise floor.** At 720m the best excess
   (0.0649%) is smaller than the *median* null spread (0.0861%). A feature
   that beats the cost floor but loses to shuffled data is not a signal.
2. **The effective sample is ~90 windows.** 45 days at a 720m horizon gives
   90 non-overlapping periods, and the overlapping windows the diagnostic
   actually uses are heavily autocorrelated. Nothing here is significant.
3. **`volatility` is not a directional signal.** Sorting bars by volatility
   and finding a forward-return spread at 12h reflects vol clustering and
   drift, not a tradeable direction.

## On STATUS's 0.6%

Not reproduced, and no claim that it was wrong — the method here differs in
two ways that both cut the number down: raw price levels are excluded, and
spreads are reported net of a shuffled-return null. Under this measurement
the numbers are 0.003%–0.15% depending on horizon, not 0.6%.

The direction of the conclusion is unaffected. If anything the case is
stronger, because it now holds against the *cheapest* floor rather than the
most expensive one.

## What this does and does not settle

Settled: **"no edge" is not an artifact of the taker cost assumption.** That
was a real bug in the cost model and it is fixed, and the conclusion
survives the fix. An agent declining to trade this remains correct
behaviour.

Not settled:

- One 45-day window (Jun–Jul 2026) in one regime. Other periods may differ.
- Quintile spread is a ceiling on a *single-feature* strategy. It does not
  rule out an edge from feature combinations, which is what the DQN was for.
- Adverse selection is still unmeasured, so `maker_adverse_selection` remains
  at its optimistic zero. The real maker floor is *higher* than 0.04%, which
  makes the negative result more robust, not less.

The honest reading: the cost model was wrong, the user was right to
challenge it, and correcting it does not change the answer.

---

# Paper trading — the trained ensemble declines to trade

Trained the full 5-agent ensemble on the 64,800 real 1m bars
(`scripts/train_ensemble.py --episodes 40`), then evaluated on the held-out
20% slice. This is the first time the system has been trained and paper
traded on real data rather than synthetic.

## Every agent independently converged on "flat"

| agent | untrained total R | trained | trades after training |
|---|---|---|---|
| momentum | −176.20 | **−4.28** | 6 |
| mean_reversion | −224.94 | **0.00** | 0 |
| volume_breakout | −354.08 | **0.00** | 0 |
| microstructure | −961.06 | **0.00** | 0 |

Ensemble on the eval slice: **0 trades, 0 fees, equity unchanged at 10,000**.

## It is deciding, not failing

The run logged **12,960 decision entries** — one per eval bar, each with the
individual agent votes recorded. A sample:

```
regime sideways | score 0.0 | direction 0
votes: momentum(0, 0.2411)  mean_reversion(0, 0.2765)
       volume_breakout(0, 0.2697)  microstructure(0, 0.2654)
```

Four agents, each evaluated the bar, each returned direction 0. A crashed or
mis-wired pipeline does not produce 12,960 audited abstentions with
per-agent confidences attached.

## What the untrained agents show it avoided

The baseline evals are the counterfactual — the same agents before learning,
trading freely:

| agent | trades | win rate | final equity from 10,000 |
|---|---|---|---|
| microstructure | 1,684 | 0.24% | **$1.82** |
| volume_breakout | 424 | 0.71% | $1,334 |
| mean_reversion | 272 | 2.57% | $2,067 |
| momentum | 143 | 5.59% | $4,471 |

Trading this market destroyed 82–99.98% of capital. Declining to trade
preserved all of it. "Stop trading" is not the agent giving up; it is the
agent finding the only non-losing action available.

## Why this matters more than the diagnostic

The cost-floor diagnostic and this run are independent methods that agree.
The diagnostic measured signal against cost analytically and found the
signal ~8x too small. The ensemble was handed capital, a full action space
and a reward for making money, and chose to sit out. Neither result depends
on the other being right.

Combined with the cost-model correction — which was a real bug, fixed, and
did not change the conclusion — the negative finding for indicator-based
trading on BTC 1m candles should now be considered well established rather
than provisional.

## Caveats that remain

- 45 days, one regime (Jun–Jul 2026). The eval slice is ~13k bars.
- 40 episodes is a short run; `config.yaml` suggests 300 for a full pass.
  More training would not plausibly *create* an edge the diagnostic says is
  absent, but the specific numbers above would move.
- `maker_adverse_selection` is still 0, so the cost floor used is the
  optimistic one. The real floor is higher, which strengthens the result.
