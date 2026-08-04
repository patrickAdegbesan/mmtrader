"""Measure the BTCUSDT spot/perp basis from Bybit's public tick archive.

This is a SMOKE TEST, not a backtest. Read research/FINDINGS.md before
drawing any conclusion from the output.

Two things it is not:

  1. The basis computed here is a trade-VWAP difference. Bybit's actual
     funding rate is derived from the time-weighted impact bid/ask against
     the index price. Trade VWAP is biased by which side is aggressing, so
     the sign here can disagree with realised funding.

  2. Annualising a single day's basis (the `ann%` column) is arithmetically
     valid but rhetorically misleading -- a 5bp day becomes "-60% APR".
     Read `basis_bp`; treat `ann%` as decoration.

Data source is https://public.bybit.com, which -- unlike api.bybit.com --
is not geo-blocked from datacenter ranges.

    python3 research/basis_probe.py 2026-01-15 2026-07-15

Perp history starts 2020-03-25; spot starts 2022-11. Their overlap is the
usable window for basis work.
"""

import gzip
import io
import statistics
import sys
import urllib.request

PERP = "https://public.bybit.com/trading/BTCUSDT/BTCUSDT%s.csv.gz"
SPOT = "https://public.bybit.com/spot/BTCUSDT/BTCUSDT_%s.csv.gz"

FUNDING_PER_DAY = 3  # Bybit settles funding every 8h
ROUND_TRIP_BP = 31   # both legs in and out, all-taker


def fetch(url):
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            return gzip.decompress(resp.read()).decode()
    except Exception:
        return None


def vwap_by_hour(text, ts_col, px_col, sz_col, ts_divisor):
    """Volume-weighted average price bucketed into hours since epoch."""
    buckets = {}
    for line in io.StringIO(text):
        fields = line.rstrip("\n").split(",")
        if not fields or fields[0] in ("timestamp", "id"):
            continue
        try:
            hour = int(float(fields[ts_col]) / ts_divisor) // 3600
            price = float(fields[px_col])
            qty = float(fields[sz_col])
        except (ValueError, IndexError):
            continue
        acc = buckets.setdefault(hour, [0.0, 0.0])
        acc[0] += price * qty
        acc[1] += qty
    return {h: notional / qty for h, (notional, qty) in buckets.items() if qty > 0}


def annualised(basis_bp):
    return basis_bp / 1e4 * FUNDING_PER_DAY * 365 * 100


def main(dates):
    print(f"{'date':12} {'spotVWAP':>10} {'perpVWAP':>10} {'basis_bp':>9} {'ann%':>8} {'hrs':>4}")
    print("-" * 60)

    daily = []
    for date in dates:
        perp_raw, spot_raw = fetch(PERP % date), fetch(SPOT % date)
        if not perp_raw or not spot_raw:
            print(f"{date:12} {'-- missing --':>30}")
            continue

        perp = vwap_by_hour(perp_raw, 0, 4, 3, 1.0)       # ts in seconds
        spot = vwap_by_hour(spot_raw, 1, 2, 3, 1000.0)    # ts in milliseconds

        hours = sorted(set(perp) & set(spot))
        if not hours:
            print(f"{date:12} no overlapping hours")
            continue

        basis_bp = statistics.fmean(
            (perp[h] - spot[h]) / spot[h] * 1e4 for h in hours
        )
        daily.append(basis_bp)
        print(
            f"{date:12} "
            f"{statistics.fmean([spot[h] for h in hours]):10.1f} "
            f"{statistics.fmean([perp[h] for h in hours]):10.1f} "
            f"{basis_bp:9.2f} "
            f"{annualised(basis_bp):8.2f} "
            f"{len(hours):4d}"
        )

    if not daily:
        return

    mean_bp = statistics.fmean(daily)
    print("-" * 60)
    print(f"mean basis {mean_bp:+.2f} bp  ->  ~{annualised(mean_bp):+.2f}% APR gross")
    print(f"days positive: {sum(1 for b in daily if b > 0)}/{len(daily)}")
    if len(daily) > 1:
        print(f"stdev across days: {statistics.stdev(daily):.2f} bp")
    print(f"\nRound-trip fee for the two-leg hedge is ~{ROUND_TRIP_BP}bp all-taker.")
    print("Weigh that against the basis column before believing in any edge.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
