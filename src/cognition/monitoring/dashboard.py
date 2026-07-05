"""File-based dashboard (Guardian B): a self-contained HTML page
regenerated periodically by the trading loop — equity curve (inline
SVG), open positions, meta-learner weights per regime, regime/session
performance, and recent trades. No server, no dependencies: open the
file in a browser (it auto-refreshes).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>Project Cognition — Paper Trading</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; background: #0d1117; color: #e6edf3; }}
h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1.05rem; margin-top: 1.5rem; color: #7ee787; }}
table {{ border-collapse: collapse; margin-top: .5rem; }}
td, th {{ border: 1px solid #30363d; padding: .3rem .6rem; font-size: .85rem; text-align: right; }}
th {{ background: #161b22; }} td:first-child, th:first-child {{ text-align: left; }}
.stat {{ display: inline-block; margin-right: 2rem; }}
.stat b {{ font-size: 1.2rem; display: block; }}
.neg {{ color: #ff7b72; }} .pos {{ color: #7ee787; }}
svg {{ background: #161b22; border: 1px solid #30363d; }}
</style></head><body>
<h1>Project Cognition — paper trading {mode}</h1>
<p>Updated {updated} · halted: <b class="{halt_class}">{halted}</b></p>
<div>
  <span class="stat"><b class="{ret_class}">{total_return}</b>total return</span>
  <span class="stat"><b>{equity}</b>equity</span>
  <span class="stat"><b>{trades}</b>closed trades</span>
  <span class="stat"><b>{win_rate}</b>win rate</span>
  <span class="stat"><b>{regime}</b>current regime</span>
</div>
<h2>Equity curve</h2>
{equity_svg}
<h2>Open positions</h2>
{positions_table}
<h2>Meta-learner weights by regime</h2>
{weights_table}
<h2>Performance by (regime, session)</h2>
{session_table}
<h2>Recent trades</h2>
{trades_table}
</body></html>
"""


def equity_svg(values: list[float], width: int = 720, height: int = 140) -> str:
    if len(values) < 2:
        return "<p>(waiting for data)</p>"
    values = values[-1000:]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    n = len(values)
    points = " ".join(
        f"{(k / (n - 1)) * (width - 10) + 5:.1f},{height - 5 - ((v - lo) / span) * (height - 10):.1f}"
        for k, v in enumerate(values)
    )
    color = "#7ee787" if values[-1] >= values[0] else "#ff7b72"
    return (
        f'<svg width="{width}" height="{height}">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/></svg>'
    )


def _table(df: pd.DataFrame, empty: str = "<p>(none)</p>") -> str:
    if df is None or df.empty:
        return empty
    return df.to_html(index=False, border=0, float_format=lambda v: f"{v:.4f}")


class DashboardWriter:
    def __init__(self, path: Path, mode: str = "replay"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode

    def write(
        self,
        *,
        equity_values: list[float],
        initial_equity: float,
        closed_trades: int,
        win_rate: float,
        current_regime: object,
        halted: bool,
        positions: pd.DataFrame,
        weights: pd.DataFrame,
        session_table: pd.DataFrame,
        recent_trades: pd.DataFrame,
    ) -> None:
        equity = equity_values[-1] if equity_values else initial_equity
        total_return = equity / initial_equity - 1
        weights_display = weights.reset_index(names="regime") if not weights.empty else weights
        html = _PAGE.format(
            mode=self.mode,
            updated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            halted=halted, halt_class="neg" if halted else "pos",
            total_return=f"{total_return:+.2%}", ret_class="neg" if total_return < 0 else "pos",
            equity=f"{equity:,.2f}",
            trades=closed_trades, win_rate=f"{win_rate:.1%}",
            regime=current_regime,
            equity_svg=equity_svg(equity_values),
            positions_table=_table(positions),
            weights_table=_table(weights_display),
            session_table=_table(session_table),
            trades_table=_table(recent_trades),
        )
        self.path.write_text(html)
