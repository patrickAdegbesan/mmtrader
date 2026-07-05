"""Markdown report assembly for backtest runs, including the explicit
pass/fail check against the project's gate-to-live criteria:
positive expectancy after costs in ALL regimes, win rate 55-65%,
Sharpe > 1.5, max drawdown < 15%.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

PASS_CRITERIA_NOTE = (
    "Pass criteria are the architecture doc's gate-to-live thresholds. "
    "A dummy strategy is EXPECTED to fail them — this run validates the "
    "plumbing (costs, walk-forward, Monte Carlo, regime split), not the edge."
)

_METRIC_FORMATS = {
    "win_rate": "{:.1%}", "total_return": "{:+.2%}", "max_drawdown": "{:.2%}",
    "expectancy_r": "{:+.3f}", "sharpe": "{:.2f}", "profit_factor": "{:.2f}",
    "avg_pnl": "{:+.2f}", "total_pnl": "{:+.2f}", "total_fees": "{:.2f}",
    "total_slippage": "{:.2f}", "avg_bars_held": "{:.1f}", "avg_mae_r": "{:.2f}",
    "initial_equity": "{:.2f}", "final_equity": "{:.2f}",
    "return_p5": "{:+.2%}", "return_p50": "{:+.2%}", "return_p95": "{:+.2%}",
    "max_dd_p50": "{:.2%}", "max_dd_p95": "{:.2%}", "prob_profitable": "{:.1%}",
}


def _fmt(key: str, value) -> str:
    if isinstance(value, float):
        template = _METRIC_FORMATS.get(key, "{:.4f}")
        return template.format(value)
    return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def metrics_section(title: str, metrics: dict) -> str:
    rows = [[k, _fmt(k, v)] for k, v in metrics.items()]
    return f"## {title}\n\n" + _md_table(["metric", "value"], rows)


def group_section(title: str, group_df: pd.DataFrame) -> str:
    if group_df.empty:
        return f"## {title}\n\n_No trades._"
    cols = ["num_trades", "win_rate", "expectancy_r", "total_pnl", "profit_factor", "total_fees"]
    cols = [c for c in cols if c in group_df.columns]
    rows = [
        [str(idx)] + [_fmt(c, group_df.loc[idx, c]) for c in cols]
        for idx in group_df.index
    ]
    return f"## {title}\n\n" + _md_table(["group"] + cols, rows)


def walkforward_section(windows) -> str:
    if not windows:
        return "## Walk-forward windows\n\n_Not enough data for any window._"
    headers = ["window", "trades", "win_rate", "expectancy_r", "return", "max_dd", "sharpe"]
    rows = []
    for w in windows:
        m = w.metrics
        rows.append([
            str(w.window), str(m["num_trades"]), _fmt("win_rate", m["win_rate"]),
            _fmt("expectancy_r", m["expectancy_r"]), _fmt("total_return", m["total_return"]),
            _fmt("max_drawdown", m["max_drawdown"]), _fmt("sharpe", m["sharpe"]),
        ])
    return "## Walk-forward windows\n\n" + _md_table(headers, rows)


def check_pass_criteria(overall: dict, regime_df: pd.DataFrame) -> list[tuple[str, str, bool]]:
    checks: list[tuple[str, str, bool]] = []

    if regime_df.empty:
        checks.append(("Positive expectancy in ALL regimes", "no trades", False))
    else:
        tradeable = regime_df[regime_df.index != "unknown"]
        all_positive = bool((tradeable["total_pnl"] > 0).all()) and not tradeable.empty
        detail = ", ".join(f"{idx}: {_fmt('total_pnl', row['total_pnl'])}" for idx, row in tradeable.iterrows())
        checks.append(("Positive expectancy in ALL regimes", detail or "no regime data", all_positive))

    wr = overall["win_rate"]
    checks.append(("Win rate 55-65%", _fmt("win_rate", wr), 0.55 <= wr <= 0.65))
    checks.append(("Sharpe > 1.5", _fmt("sharpe", overall["sharpe"]), overall["sharpe"] > 1.5))
    checks.append(("Max drawdown < 15%", _fmt("max_drawdown", overall["max_drawdown"]), overall["max_drawdown"] < 0.15))
    return checks


def pass_criteria_section(checks: list[tuple[str, str, bool]]) -> str:
    rows = [[name, value, "PASS" if ok else "FAIL"] for name, value, ok in checks]
    return (
        "## Gate-to-live pass criteria\n\n"
        + _md_table(["criterion", "value", "status"], rows)
        + f"\n\n_{PASS_CRITERIA_NOTE}_"
    )


def build_report(title: str, sections: list[str]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"# {title}\n\n_Generated {stamp}_\n"
    return "\n\n".join([header] + sections) + "\n"
