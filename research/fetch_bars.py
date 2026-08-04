"""Download Bybit perp ticks and aggregate to 1m OHLCV bars."""
import gzip, io, urllib.request, csv, sys, datetime as dt

URL = "https://public.bybit.com/trading/BTCUSDT/BTCUSDT%s.csv.gz"

def day_bars(date):
    try:
        with urllib.request.urlopen(URL % date, timeout=300) as r:
            raw = gzip.decompress(r.read()).decode()
    except Exception as e:
        print(f"  {date}: FAIL {e}", flush=True); return []
    bars = {}
    for row in csv.reader(io.StringIO(raw)):
        if not row or row[0] == "timestamp": continue
        try:
            ts = float(row[0]); px = float(row[4]); qty = float(row[3])
        except (ValueError, IndexError): continue
        m = int(ts) // 60
        b = bars.get(m)
        if b is None:
            bars[m] = [px, px, px, px, qty]          # o h l c v
        else:
            if px > b[1]: b[1] = px
            if px < b[2]: b[2] = px
            b[3] = px; b[4] += qty
    print(f"  {date}: {len(bars)} bars", flush=True)
    return [(m*60, *v) for m, v in sorted(bars.items())]

start = dt.date.fromisoformat(sys.argv[1]); n = int(sys.argv[2])
out = []
for i in range(n):
    out += day_bars((start + dt.timedelta(days=i)).isoformat())
with open("bars_1m.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["timestamp","open","high","low","close","volume"]); w.writerows(out)
print(f"TOTAL {len(out)} bars -> bars_1m.csv", flush=True)
