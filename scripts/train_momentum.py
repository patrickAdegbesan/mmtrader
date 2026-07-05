#!/usr/bin/env python3
"""One-command entrypoint for Milestone 3: train the momentum agent
end-to-end, prove the learning loop improves it, and validate the
trained policy through the Milestone-2 backtester.

Usage:
    python scripts/train_momentum.py                                  # synthetic demo
    python scripts/train_momentum.py --data data/processed/BTC_USDT_1m.parquet
    python scripts/train_momentum.py --episodes 100 --synthetic-bars 40000

Outputs:
- learning curve (baseline untrained eval -> periodic greedy evals)
- versioned checkpoints under models/momentum/ (rollback: activate an
  older version)
- a cost-inclusive backtest report of the trained agent on the eval slice
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
from cognition.agents.momentum import DQNStrategy, load_momentum_agent  # noqa: E402
from cognition.agents.trainer import AgentTrainer, TrainerConfig  # noqa: E402
from cognition.backtest.costs import CostModel  # noqa: E402
from cognition.backtest.metrics import compute_metrics  # noqa: E402
from cognition.backtest.simulator import EventDrivenBacktester  # noqa: E402
from cognition.backtest.synthetic import make_synthetic_ohlcv  # noqa: E402
from cognition.features.engineering import extract_features  # noqa: E402
from cognition.learning.registry import ModelRegistry  # noqa: E402
from cognition.utils.logging import get_logger, log_with_fields  # noqa: E402
from cognition.utils.timeframes import bars_per_year  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--synthetic-bars", type=int, default=20_000)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, _ = get_settings()
    tc = config.training
    bt = config.backtest
    logger = get_logger("train_momentum", log_dir=config.log_dir)

    if args.data:
        df = pd.read_parquet(args.data)
        source = args.data
    else:
        df = make_synthetic_ohlcv(n_bars=args.synthetic_bars, timeframe=args.timeframe)
        source = f"synthetic ({args.synthetic_bars} bars)"
    if "volatility" not in df.columns:
        df = extract_features(df)
    log_with_fields(logger, 20, "Training data ready", source=source, bars=len(df))

    cost_model = CostModel(
        taker_fee=bt.taker_fee, slippage_base=bt.slippage_base,
        slippage_high_vol=bt.slippage_high_vol, latency_bars=bt.latency_bars,
        fill_ratio=bt.fill_ratio,
    )
    mapper = ActionMapper(
        vol_stop_scale=tc.vol_stop_scale, min_stop_pct=tc.min_stop_pct,
        max_stop_pct=tc.max_stop_pct, reward_risk=tc.reward_risk,
    )
    registry = ModelRegistry(config.models_dir, "momentum")
    trainer_config = TrainerConfig(
        episodes=args.episodes or tc.episodes,
        episode_bars=tc.episode_bars,
        eval_every=tc.eval_every,
        eval_fraction=tc.eval_fraction,
        seed=args.seed if args.seed is not None else tc.seed,
        dqn=DQNConfig(seed=args.seed if args.seed is not None else tc.seed),
    )
    trainer = AgentTrainer(df, cost_model, mapper, registry, trainer_config)

    history = trainer.train()

    print("\n=== Learning curve (greedy eval on held-out slice) ===")
    print(f"{'episode':>8} {'total_R':>10} {'trades':>7} {'avg_R':>8} {'win_rate':>9} {'equity':>10}")
    for h in history:
        print(f"{h.episode:>8} {h.total_reward:>10.2f} {h.num_trades:>7} {h.avg_r:>8.3f} {h.win_rate:>9.1%} {h.final_equity:>10.2f}")

    baseline, final = history[0], history[-1]
    improved = final.total_reward > baseline.total_reward
    print(f"\nBaseline (untrained) total R: {baseline.total_reward:.2f}")
    print(f"Final (trained)      total R: {final.total_reward:.2f}")
    print(f"Learning loop improved the agent: {'YES' if improved else 'NO'}")
    print(f"Model versions saved: {registry.list_versions()} (LATEST -> {registry.latest_version()})")

    # --- Validate the trained agent through the M2 backtester -----------
    agent, stats, _ = load_momentum_agent(registry)
    strategy = DQNStrategy(agent, stats, mapper)
    engine = EventDrivenBacktester(
        cost_model, initial_equity=bt.initial_equity,
        risk_per_trade=bt.risk_per_trade, max_position_fraction=bt.max_position_fraction,
    )
    result = engine.run(trainer.eval_df, strategy)
    metrics = compute_metrics(result, bars_per_year(args.timeframe))
    print("\n=== Trained agent through the backtester (eval slice, after all costs) ===")
    for key in ("num_trades", "win_rate", "expectancy_r", "total_return", "sharpe", "max_drawdown", "total_fees"):
        print(f"  {key}: {metrics[key]:.4f}" if isinstance(metrics[key], float) else f"  {key}: {metrics[key]}")

    return 0 if improved else 1


if __name__ == "__main__":
    raise SystemExit(main())
