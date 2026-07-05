#!/usr/bin/env python3
"""One-command entrypoint for Milestone 2: run a cost-realistic backtest
of the dummy SMA-crossover strategy with walk-forward validation, Monte
Carlo robustness testing, regime-split reporting, and the 6-month
out-of-sample lockbox excluded.

Usage:
    python scripts/run_backtest.py                     # synthetic data demo
    python scripts/run_backtest.py --data data/processed/BTC_USDT_1m.parquet
    python scripts/run_backtest.py --synthetic-bars 60000 --mc-runs 100
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cognition.config import get_settings  # noqa: E402
from cognition.backtest.costs import CostModel  # noqa: E402
from cognition.backtest.metrics import compute_metrics, stats_by_group, trades_to_frame  # noqa: E402
from cognition.backtest.montecarlo import run_monte_carlo, summarize_monte_carlo  # noqa: E402
from cognition.backtest.regimes import label_market_regimes  # noqa: E402
from cognition.backtest.report import (  # noqa: E402
    build_report, check_pass_criteria, group_section, metrics_section,
    pass_criteria_section, walkforward_section,
)
from cognition.backtest.simulator import EventDrivenBacktester  # noqa: E402
from cognition.backtest.strategies import SmaCrossoverStrategy  # noqa: E402
from cognition.backtest.synthetic import make_synthetic_ohlcv  # noqa: E402
from cognition.backtest.walkforward import run_walk_forward, split_lockbox  # noqa: E402
from cognition.features.engineering import extract_features  # noqa: E402
from cognition.utils.logging import get_logger, log_with_fields  # noqa: E402
from cognition.utils.timeframes import bars_per_day, bars_per_year  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="Path to a processed parquet file")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--synthetic-bars", type=int, default=30_000)
    parser.add_argument("--mc-runs", type=int, default=None, help="Override Monte Carlo run count")
    parser.add_argument("--mc-bars", type=int, default=10_000, help="Bars of recent data used for Monte Carlo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = get_settings()
    bt = config.backtest
    logger = get_logger("run_backtest", log_dir=config.log_dir)

    # --- Load or synthesize data -------------------------------------
    if args.data:
        df = pd.read_parquet(args.data)
        source = args.data
    else:
        df = make_synthetic_ohlcv(n_bars=args.synthetic_bars, timeframe=args.timeframe)
        source = f"synthetic ({args.synthetic_bars} bars)"
    if "volatility_regime" not in df.columns:
        df = extract_features(df)

    bpd = bars_per_day(args.timeframe)
    bpy = bars_per_year(args.timeframe)
    regime_window = min(bt.regime_window_days * bpd, max(len(df) // 10, 50))
    df = label_market_regimes(df, window_bars=regime_window, threshold=bt.regime_threshold)

    # --- Lockbox: recent months split off, never touched below -------
    main_df, lockbox_df = split_lockbox(df, months=bt.lockbox_months)
    if len(main_df) < 1_000:
        # Short datasets (synthetic demos) would lose everything to the
        # lockbox; keep the split honest for real 5-year data.
        main_df, lockbox_df = df, df.iloc[0:0]
    log_with_fields(
        logger, 20, "Data prepared",
        source=source, total_bars=len(df), main_bars=len(main_df), lockbox_bars=len(lockbox_df),
    )

    cost_model = CostModel(
        taker_fee=bt.taker_fee,
        slippage_base=bt.slippage_base,
        slippage_high_vol=bt.slippage_high_vol,
        latency_bars=bt.latency_bars,
        fill_ratio=bt.fill_ratio,
    )
    engine = EventDrivenBacktester(
        cost_model,
        initial_equity=bt.initial_equity,
        risk_per_trade=bt.risk_per_trade,
        max_position_fraction=bt.max_position_fraction,
    )
    strategy_factory = SmaCrossoverStrategy

    # --- Full-sample run on the main (non-lockbox) data --------------
    result = engine.run(main_df, strategy_factory())
    overall = compute_metrics(result, bpy)
    trades_df = trades_to_frame(result.trades)
    regime_stats = stats_by_group(trades_df, "market_regime")
    session_stats = stats_by_group(trades_df, "session")
    log_with_fields(logger, 20, "Full-sample backtest done", **{k: overall[k] for k in ("num_trades", "total_return", "win_rate", "sharpe", "max_drawdown")})

    # --- Walk-forward -------------------------------------------------
    train_bars = min(bt.walkforward.train_days * bpd, len(main_df) // 3)
    test_bars = min(bt.walkforward.test_days * bpd, max(len(main_df) // 6, 1))
    windows, wf_trades = run_walk_forward(
        main_df, engine, strategy_factory,
        train_bars=train_bars, test_bars=test_bars,
        warmup_bars=bt.walkforward.warmup_bars, bars_per_year=bpy,
    )
    log_with_fields(logger, 20, "Walk-forward done", windows=len(windows), train_bars=train_bars, test_bars=test_bars)

    # --- Monte Carlo on the most recent slice -------------------------
    mc_runs = args.mc_runs if args.mc_runs is not None else bt.montecarlo.runs
    mc_slice = main_df.iloc[-args.mc_bars:].reset_index(drop=True)
    mc_df = run_monte_carlo(
        mc_slice, engine, strategy_factory,
        runs=mc_runs, bars_per_year=bpy, noise_scale=bt.montecarlo.noise_scale,
    )
    mc_summary = summarize_monte_carlo(mc_df)
    log_with_fields(logger, 20, "Monte Carlo done", **mc_summary)

    # --- Report --------------------------------------------------------
    cost_summary = {
        "taker_fee": bt.taker_fee, "slippage_base": bt.slippage_base,
        "slippage_high_vol": bt.slippage_high_vol, "latency_bars": bt.latency_bars,
        "fill_ratio": bt.fill_ratio, "risk_per_trade": bt.risk_per_trade,
        "total_fees_paid": overall["total_fees"], "total_slippage_paid": overall["total_slippage"],
    }
    checks = check_pass_criteria(overall, regime_stats)
    report_md = build_report(
        f"Backtest report — SMA crossover (dummy) — {source}",
        [
            metrics_section("Cost model (applied to every fill)", cost_summary),
            metrics_section("Overall (full sample, after all costs)", overall),
            group_section("By market regime (bull / bear / sideways)", regime_stats),
            group_section("By session (Asian / European / US)", session_stats),
            walkforward_section(windows),
            metrics_section("Monte Carlo robustness", mc_summary or {"runs": 0}),
            pass_criteria_section(checks),
            f"## Lockbox\n\nMost recent {bt.lockbox_months} months "
            f"({len(lockbox_df)} bars) excluded from everything above; reserved for final validation.",
        ],
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = config.reports_dir / f"backtest_{stamp}.md"
    out_path.write_text(report_md)

    print(report_md)
    print(f"Report saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
