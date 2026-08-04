"""Does the feature signal clear its cost floor -- and does the answer change
when the floor is priced for the maker execution the executor actually uses?

docs/STATUS.md records "no edge" from a feature/label diagnostic measuring a
~0.6% quintile spread against a 0.4%-1.2% round-trip floor. That floor is the
taker floor. `execution/executor.py` quotes maker-first. So the recorded
conclusion may be an artifact of pricing a strategy the system does not run.

This re-runs the diagnostic and reports the spread against both floors.

Method, and its limits:

  For each feature, bucket bars into quintiles and measure the mean forward
  return over `horizon` bars. The spread between the best and worst quintile
  is the gross signal available to a strategy that could perfectly sort on
  that feature alone. That is generous -- it assumes you trade only the tails
  and always pick the right one -- so treat it as a ceiling on the edge, not
  an estimate of it. A signal that fails to clear its cost floor here fails
  a fortiori in a real strategy.

  Forward returns are computed from bar close to bar close `horizon` bars
  ahead, with no execution modelled. Costs are taken from CostModel.

## The null baseline, and why it is not optional

Run this without a control and it lies. On a synthetic random walk -- no
edge by construction -- the first version of this script reported sma_9
clearing the maker floor at 1.96x and 5 of 19 features "passing".

Two things produce that. Raw price levels (sma_9, ema_21, the moving
averages themselves) are not signals: quintiling a dollar price over a
period when price drifted sorts bars by *when they happened*, so the
"spread" is the drift. Those are excluded below. And any feature shows some
spread by chance, so a raw number means nothing without knowing what chance
looks like.

So every run also measures a surrogate: the same bars with their returns
shuffled, which destroys temporal structure while preserving the return
distribution exactly. Features are recomputed on that. Whatever spread
survives is what this method manufactures from noise. Only the excess over
that baseline is evidence of anything.

Usage:

    python3 research/cost_floor_diagnostic.py bars_1m.csv [horizon]

Bars are 1m OHLCV [timestamp, open, high, low, close, volume]; build them
with research/fetch_bars.py from Bybit's public tick archive.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from cognition.backtest.costs import CostModel
from cognition.features.engineering import extract_features

# Bybit VIP0. Spot charges both sides 0.1%; perps split 0.055/0.02, which is
# where the maker saving actually comes from.
PERP = CostModel(taker_fee=0.00055, maker_fee=0.0002, slippage_base=0.001)
SPOT = CostModel(taker_fee=0.001, maker_fee=0.001, slippage_base=0.001)

# Columns that are labels, bookkeeping or non-numeric rather than signals.
NOT_FEATURES = {
    "timestamp", "open", "high", "low", "close", "volume",
    "volatility_regime", "session", "fwd_return",
}

# Absolute price levels in dollars. Not signals -- quintiling them sorts by
# where price happened to be, so on any trending sample they report the drift
# as edge. The distance/position features derived from them are fine; these
# raw levels are not.
PRICE_LEVELS = {"sma_9", "sma_21", "ema_9", "ema_21"}

QUINTILES = 5
NULL_SEED = 0


def quintile_spread(df: pd.DataFrame, col: str) -> tuple[float, float, float] | None:
    """Best-minus-worst mean forward return across quintiles of `col`."""
    sub = df[[col, "fwd_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < QUINTILES * 100 or sub[col].nunique() < QUINTILES:
        return None
    try:
        buckets = pd.qcut(sub[col], QUINTILES, labels=False, duplicates="drop")
    except ValueError:
        return None
    means = sub.groupby(buckets)["fwd_return"].mean()
    if len(means) < 2:
        return None
    return float(means.max() - means.min()), float(means.max()), float(means.min())


def shuffle_returns(df: pd.DataFrame, seed: int = NULL_SEED) -> pd.DataFrame:
    """Surrogate series: same returns, random order.

    Preserves the return distribution (so volatility, fat tails and the
    resulting feature scales all match) while destroying the temporal
    structure any real signal would live in. Spread measured here is the
    method's false-positive rate, not an edge.
    """
    rng = np.random.default_rng(seed)
    rets = df["close"].pct_change().dropna().to_numpy(copy=True)
    rng.shuffle(rets)

    close = df["close"].iloc[0] * np.cumprod(np.r_[1.0, 1.0 + rets])
    scale = close / df["close"].to_numpy()

    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = df[col].to_numpy() * scale
    return out


def spreads_for(df: pd.DataFrame, horizon: int) -> dict[str, tuple[float, float, float]]:
    feat = extract_features(df.copy())
    feat["fwd_return"] = feat["close"].shift(-horizon) / feat["close"] - 1.0
    cols = [
        c for c in feat.columns
        if c not in NOT_FEATURES
        and c not in PRICE_LEVELS
        and pd.api.types.is_numeric_dtype(feat[c])
    ]
    out = {}
    for c in cols:
        got = quintile_spread(feat, c)
        if got:
            out[c] = got
    return out


def main(path: str, horizon: int) -> None:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    real = spreads_for(df, horizon)
    null = spreads_for(shuffle_returns(df), horizon)

    rows = [
        (name, spread, best, worst, null.get(name, (0.0,))[0])
        for name, (spread, best, worst) in real.items()
    ]
    rows.sort(key=lambda r: -(r[1] - r[4]))

    taker = PERP.round_trip_cost("low", "taker")
    maker = PERP.round_trip_cost("low", "maker")
    spot_taker = SPOT.round_trip_cost("low", "taker")

    print(f"bars {len(df):,}  "
          f"{df['timestamp'].iloc[0]:%Y-%m-%d} -> {df['timestamp'].iloc[-1]:%Y-%m-%d}  "
          f"horizon {horizon} bars")
    print()
    print("cost floors (round trip, fraction of price)")
    print(f"  perp taker  {taker:.5f}  ({taker*100:.3f}%)   <- what the backtest charges")
    print(f"  perp maker  {maker:.5f}  ({maker*100:.3f}%)   <- what the executor aims for")
    print(f"  spot taker  {spot_taker:.5f}  ({spot_taker*100:.3f}%)   <- current config category")
    print()
    if not rows:
        print("no features produced a usable quintile spread")
        return

    print(f"{'feature':24} {'spread%':>9} {'null%':>8} {'excess%':>9} "
          f"{'vs taker':>9} {'vs maker':>9}")
    print("-" * 76)
    for name, spread, _best, _worst, null_spread in rows:
        excess = spread - null_spread
        print(f"{name:24} {spread*100:9.4f} {null_spread*100:8.4f} {excess*100:9.4f} "
              f"{excess/taker:9.2f}x {excess/maker:9.2f}x")

    beats_taker = [r for r in rows if (r[1] - r[4]) > taker]
    beats_maker = [r for r in rows if (r[1] - r[4]) > maker]
    best_name, best_spread, _, _, best_null = rows[0]
    best_excess = best_spread - best_null

    print("-" * 76)
    print(f"excess over null clearing perp taker floor: {len(beats_taker)}/{len(rows)}")
    print(f"excess over null clearing perp maker floor: {len(beats_maker)}/{len(rows)}")
    print(f"best excess {best_excess*100:.4f}% ({best_name}: "
          f"{best_spread*100:.4f}% real - {best_null*100:.4f}% null)")
    print(f"median null spread {np.median([r[4] for r in rows])*100:.4f}% "
          f"<- what this method invents from noise")
    print()
    print("Read the excess column, not the spread column. Spread alone counts")
    print("the method's own false positives as edge.")
    print()
    print("And excess is still a ceiling, not an edge: it assumes perfect")
    print("sorting into the tails and models no execution. Clearing the maker")
    print("floor here is necessary for a strategy to work, nowhere near")
    print("sufficient -- adverse selection is not in these numbers.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
