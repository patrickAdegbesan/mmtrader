#!/usr/bin/env python3
"""Scheduled full retraining (Milestone 7). Run nightly/weekly via cron,
a systemd timer, or the `retrainer` service in docker-compose.

For every agent: warm-start from the production version, retrain on the
provided data, evaluate old vs new on the same held-out slice, and
promote ONLY if the new version doesn't underperform. A worse version is
kept on disk (inspectable) but never activated — no silent updates ever.

Usage:
    python scripts/retrain.py --data data/processed/BTC_USDT_1m.parquet
    python scripts/retrain.py --synthetic-bars 20000 --episodes 30   # smoke test
    python scripts/retrain.py ... --tolerance 50    # allow small regressions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cognition.config import get_settings  # noqa: E402
from cognition.agents.actions import ActionMapper  # noqa: E402
from cognition.agents.dqn import DQNConfig  # noqa: E402
from cognition.agents.specs import AGENT_SPECS  # noqa: E402
from cognition.agents.trainer import TrainerConfig  # noqa: E402
from cognition.backtest.costs import CostModel  # noqa: E402
from cognition.backtest.synthetic import make_synthetic_ohlcv  # noqa: E402
from cognition.features.engineering import extract_features  # noqa: E402
from cognition.learning.registry import ModelRegistry  # noqa: E402
from cognition.learning.retrain import retrain_agent  # noqa: E402
from cognition.monitoring.alerts import TelegramAlerter  # noqa: E402
from cognition.utils.logging import get_logger, log_with_fields  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--synthetic-bars", type=int, default=20_000)
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Equity-units regression allowed before refusing promotion")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, secrets = get_settings()
    tc, bt = config.training, config.backtest
    logger = get_logger("retrain_cli", log_dir=config.log_dir)

    if args.data:
        df = pd.read_parquet(args.data)
        source = args.data
    else:
        df = make_synthetic_ohlcv(n_bars=args.synthetic_bars, timeframe=args.timeframe)
        source = f"synthetic ({args.synthetic_bars} bars)"
    if "volatility" not in df.columns or "range_pct" not in df.columns:
        df = extract_features(df)
    log_with_fields(logger, 20, "Retraining started", source=source, bars=len(df))

    cost_model = CostModel(
        taker_fee=bt.taker_fee, slippage_base=bt.slippage_base,
        slippage_high_vol=bt.slippage_high_vol, latency_bars=bt.latency_bars,
        fill_ratio=bt.fill_ratio,
    )
    mapper = ActionMapper(
        vol_stop_scale=tc.vol_stop_scale, min_stop_pct=tc.min_stop_pct,
        max_stop_pct=tc.max_stop_pct, reward_risk=tc.reward_risk,
    )
    alerter = TelegramAlerter(token=secrets.telegram_bot_token, chat_id=secrets.telegram_chat_id)

    seed = args.seed if args.seed is not None else tc.seed
    results = []
    for spec in AGENT_SPECS.values():
        result = retrain_agent(
            df, spec, ModelRegistry(config.models_dir, spec.name), cost_model, mapper,
            TrainerConfig(
                episodes=args.episodes or tc.episodes, episode_bars=tc.episode_bars,
                eval_every=max((args.episodes or tc.episodes) // 2, 1),
                eval_fraction=tc.eval_fraction, seed=seed, dqn=DQNConfig(seed=seed),
            ),
            tolerance=args.tolerance,
        )
        results.append(result)

    print(f"\n=== Retraining report ({source}) ===")
    print(f"{'agent':18} {'new ver':>8} {'promoted':>9} {'prod score':>12} {'new score':>11} {'active':>8}")
    demoted = []
    for r in results:
        old = f"{r.old_score:.2f}" if r.old_score is not None else "—"
        print(f"{r.agent_name:18} {r.new_version:>8} {str(r.promoted):>9} {old:>12} {r.new_score:>11.2f} {r.active_version:>8}")
        if not r.promoted:
            demoted.append(r)

    if demoted:
        names = ", ".join(r.agent_name for r in demoted)
        alerter.alert("retrain_rollback",
                      f"Retraining refused promotion for: {names}. Production versions stay active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
