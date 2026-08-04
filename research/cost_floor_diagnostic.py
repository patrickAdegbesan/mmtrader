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

QUINTILES = 5


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


def main(path: str, horizon: int) -> None:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    feat = extract_features(df)
    feat["fwd_return"] = feat["close"].shift(-horizon) / feat["close"] - 1.0

    cols = [
        c for c in feat.columns
        if c not in NOT_FEATURES and pd.api.types.is_numeric_dtype(feat[c])
    ]

    rows = []
    for c in cols:
        got = quintile_spread(feat, c)
        if got:
            rows.append((c, *got))
    rows.sort(key=lambda r: -r[1])

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
    print(f"{'feature':24} {'spread%':>9} {'best%':>8} {'worst%':>8}  {'vs taker':>9} {'vs maker':>9}")
    print("-" * 76)
    for name, spread, best, worst in rows:
        print(f"{name:24} {spread*100:9.4f} {best*100:8.4f} {worst*100:8.4f} "
              f"{spread/taker:9.2f}x {spread/maker:9.2f}x")

    if not rows:
        print("no features produced a usable quintile spread")
        return

    beats_taker = [r for r in rows if r[1] > taker]
    beats_maker = [r for r in rows if r[1] > maker]
    print("-" * 76)
    print(f"features clearing perp taker floor: {len(beats_taker)}/{len(rows)}")
    print(f"features clearing perp maker floor: {len(beats_maker)}/{len(rows)}")
    print(f"best spread {rows[0][1]*100:.4f}% ({rows[0][0]})")
    print()
    print("Reminder: the spread is a ceiling, not an edge. It assumes perfect")
    print("sorting into the tails and models no execution. Clearing the maker")
    print("floor here is necessary for a strategy to work, nowhere near")
    print("sufficient -- adverse selection is not in these numbers.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 15)
