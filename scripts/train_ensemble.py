#!/usr/bin/env python3
"""One-command entrypoint for Milestone 4: train all four market agents,
assemble the 5-agent ensemble (4 agents + meta-learner), and run it
through the cost-inclusive backtester with the meta-learner updating
online after every closed trade.

Usage:
    python scripts/train_ensemble.py                                # synthetic demo
    python scripts/train_ensemble.py --data data/processed/BTC_USDT_1m.parquet
    python scripts/train_ensemble.py --episodes 60

Outputs:
- one versioned model per agent under models/<agent_name>/
- meta-learner weights per market regime (the regime memory, learned)
- per-(regime, session) performance table
- cost-inclusive metrics for the ensemble on the eval slice
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
from cognition.agents.ensemble import EnsembleMember, EnsembleStrategy  # noqa: E402
from cognition.agents.meta import MetaLearner  # noqa: E402
from cognition.agents.momentum import load_agent  # noqa: E402
from cognition.agents.specs import AGENT_SPECS  # noqa: E402
from cognition.agents.trainer import AgentTrainer, TrainerConfig  # noqa: E402
from cognition.backtest.costs import CostModel  # noqa: E402
from cognition.backtest.metrics import compute_metrics  # noqa: E402
from cognition.backtest.regimes import label_market_regimes  # noqa: E402
from cognition.backtest.simulator import EventDrivenBacktester  # noqa: E402
from cognition.backtest.synthetic import make_synthetic_ohlcv  # noqa: E402
from cognition.features.engineering import extract_features  # noqa: E402
from cognition.learning.registry import ModelRegistry  # noqa: E402
from cognition.utils.logging import get_logger, log_with_fields  # noqa: E402
from cognition.utils.timeframes import bars_per_day, bars_per_year  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--synthetic-bars", type=int, default=20_000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vote-threshold", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = get_settings()
    tc, bt = config.training, config.backtest
    logger = get_logger("train_ensemble", log_dir=config.log_dir)

    if args.data:
        df = pd.read_parquet(args.data)
        source = args.data
    else:
        df = make_synthetic_ohlcv(n_bars=args.synthetic_bars, timeframe=args.timeframe)
        source = f"synthetic ({args.synthetic_bars} bars)"
    if "volatility" not in df.columns or "range_pct" not in df.columns:
        df = extract_features(df)
    regime_window = min(bt.regime_window_days * bars_per_day(args.timeframe), max(len(df) // 10, 50))
    df = label_market_regimes(df, window_bars=regime_window, threshold=bt.regime_threshold)
    log_with_fields(logger, 20, "Ensemble training data ready", source=source, bars=len(df))

    cost_model = CostModel(
        taker_fee=bt.taker_fee, slippage_base=bt.slippage_base,
        slippage_high_vol=bt.slippage_high_vol, latency_bars=bt.latency_bars,
        fill_ratio=bt.fill_ratio,
    )
    mapper = ActionMapper(
        vol_stop_scale=tc.vol_stop_scale, min_stop_pct=tc.min_stop_pct,
        max_stop_pct=tc.max_stop_pct, reward_risk=tc.reward_risk,
    )

    # --- Train each of the four market agents on its own feature lens ---
    members: list[EnsembleMember] = []
    eval_df = None
    for spec in AGENT_SPECS.values():
        registry = ModelRegistry(config.models_dir, spec.name)
        trainer = AgentTrainer(
            df, cost_model, mapper, registry,
            TrainerConfig(
                episodes=args.episodes, episode_bars=tc.episode_bars,
                eval_every=max(args.episodes // 2, 1), eval_fraction=tc.eval_fraction,
                seed=args.seed, dqn=DQNConfig(seed=args.seed),
            ),
            feature_columns=spec.feature_columns,
        )
        history = trainer.train()
        baseline, final = history[0], history[-1]
        log_with_fields(
            logger, 20, "Agent trained",
            agent=spec.name, baseline_r=round(baseline.total_reward, 2),
            final_r=round(final.total_reward, 2), version=registry.latest_version(),
        )
        print(f"[{spec.name:16s}] eval total R: {baseline.total_reward:9.2f} (untrained) -> {final.total_reward:9.2f} (trained)")
        agent, stats, _ = load_agent(registry)
        members.append(EnsembleMember(spec.name, agent, stats, spec.feature_columns))
        eval_df = trainer.eval_df  # identical split across agents (same df & fraction)

    # --- Assemble and run the ensemble on the eval slice -----------------
    meta = MetaLearner([m.name for m in members])
    strategy = EnsembleStrategy(members, mapper, meta=meta, vote_threshold=args.vote_threshold)
    engine = EventDrivenBacktester(
        cost_model, initial_equity=bt.initial_equity,
        risk_per_trade=bt.risk_per_trade, max_position_fraction=bt.max_position_fraction,
    )
    result = engine.run(eval_df, strategy)
    metrics = compute_metrics(result, bars_per_year(args.timeframe))

    print("\n=== Ensemble on eval slice (after all costs, meta learning online) ===")
    for key in ("num_trades", "win_rate", "expectancy_r", "total_return", "sharpe", "max_drawdown", "total_fees"):
        value = metrics[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"  meta-learner online updates: {meta.updates}")

    print("\n=== Meta-learner weights by market regime (regime memory) ===")
    snapshot = meta.snapshot()
    print(snapshot.round(3).to_string() if not snapshot.empty else "  (no regimes seen)")

    print("\n=== Performance by (regime, session) ===")
    table = strategy.session_tracker.table()
    print(table.round(3).to_string(index=False) if not table.empty else "  (no trades)")

    print(f"\nDecision log entries (auditable votes): {len(strategy.decision_log)}")
    if strategy.decision_log:
        sample = strategy.decision_log[-1]
        print(f"Sample decision: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
