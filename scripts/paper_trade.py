#!/usr/bin/env python3
"""One-command entrypoint for Milestone 6: paper trading with the full
5-agent ensemble, Guardian A + B, the trade journal, and real-time
learning.

Modes:
    # Replay (works anywhere — plays historical/synthetic data as if live):
    python scripts/paper_trade.py --replay-synthetic 3000
    python scripts/paper_trade.py --replay data/processed/BTC_USDT_1m.parquet

    # Live Bybit TESTNET data via REST polling (needs network access):
    python scripts/paper_trade.py --live

    # Live via WebSocket (lowest latency; needs WS-capable network):
    python scripts/paper_trade.py --live --ws

Agents load from models/<name>/ registries (train via
scripts/train_ensemble.py first) or use fresh untrained agents with
--fresh-agents for plumbing checks. Telegram alerts fire if
TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID are set in .env; otherwise alerts go
to the structured log. Watch reports/dashboard.html while it runs.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from cognition.config import get_settings  # noqa: E402
from cognition.agents.actions import N_ACTIONS, ActionMapper  # noqa: E402
from cognition.agents.dqn import DQNAgent, DQNConfig  # noqa: E402
from cognition.agents.ensemble import EnsembleMember, EnsembleStrategy  # noqa: E402
from cognition.agents.env import FeatureStats  # noqa: E402
from cognition.agents.momentum import load_agent  # noqa: E402
from cognition.agents.specs import AGENT_SPECS  # noqa: E402
from cognition.backtest.costs import CostModel  # noqa: E402
from cognition.backtest.synthetic import make_synthetic_ohlcv  # noqa: E402
from cognition.data.bybit_client import BybitClient  # noqa: E402
from cognition.data.feed import ReplayFeed, RestPollingFeed  # noqa: E402
from cognition.features.engineering import extract_features  # noqa: E402
from cognition.learning.journal import TradeJournal  # noqa: E402
from cognition.learning.registry import ModelRegistry  # noqa: E402
from cognition.monitoring.alerts import TelegramAlerter  # noqa: E402
from cognition.monitoring.dashboard import DashboardWriter  # noqa: E402
from cognition.paper.trader import PaperTrader  # noqa: E402
from cognition.risk.engine import RiskEngine, RiskLimits  # noqa: E402
from cognition.utils.logging import configure_file_logging  # noqa: E402
from cognition.utils.timeframes import timeframe_to_ms  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=str, nargs="*", help="Parquet file(s) to replay")
    parser.add_argument("--replay-synthetic", type=int, default=None, help="Replay N synthetic bars")
    parser.add_argument("--live", action="store_true", help="Live Bybit TESTNET data")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket feed (with --live)")
    parser.add_argument("--symbols", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--fresh-agents", action="store_true", help="Untrained agents (plumbing checks)")
    parser.add_argument("--max-bars", type=int, default=None)
    return parser.parse_args()


def build_members(fresh: bool, models_dir) -> list[EnsembleMember]:
    members = []
    for spec in AGENT_SPECS.values():
        if fresh:
            agent = DQNAgent(len(spec.feature_columns) + 3, N_ACTIONS, DQNConfig(warmup_steps=32, batch_size=16))
            stats = None  # fitted on first prepared frame below
        else:
            registry = ModelRegistry(models_dir, spec.name)
            agent, stats, _ = load_agent(registry)
        members.append(EnsembleMember(spec.name, agent, stats, spec.feature_columns))
    return members


def main() -> int:
    args = parse_args()
    config, secrets = get_settings()
    # Filename matches the Docker healthcheck (docker-compose.yml), which
    # watches this file's mtime as the liveness signal. Every cognition.*
    # module active during the run (feed, risk engine, executor, alerts,
    # paper_trader itself) now writes here, so a stalled loop of ANY kind
    # is what trips the healthcheck, not just the paper_trader logger.
    configure_file_logging(config.log_dir, "paper_trader")
    pc, bt, tc, rc = config.paper, config.backtest, config.training, config.risk

    symbols = [s.strip() for s in args.symbols.split(",")]
    tf_ms = timeframe_to_ms(args.timeframe)

    # --- feed -----------------------------------------------------------
    if args.live:
        # Safety interlock stays on the ENV, not on the data client: with
        # BYBIT_ENV=testnet no code path in this process can reach a live
        # order endpoint.
        if secrets.bybit_env != "testnet":
            print("Refusing: BYBIT_ENV must be testnet for paper trading.")
            return 1
        # ...but the market data itself must be REAL. Paper trading
        # simulates its own fills and never sends an order, so it needs
        # mainnet prices; testnet candles would make the whole campaign
        # fictional and its results meaningless.
        data_client = BybitClient(config, secrets, public_data_only=True)
        if args.ws:
            from cognition.data.websocket_feed import BybitWebSocketFeed
            feed = BybitWebSocketFeed(symbols, args.timeframe, testnet=False)
            feed.start()
        else:
            feed = RestPollingFeed(data_client, symbols, args.timeframe)
        mode, realtime = "live-data-paper", True
    else:
        frames: dict[str, pd.DataFrame] = {}
        if args.replay:
            for path in args.replay:
                symbol = Path(path).stem.replace("_1m", "").replace("_5m", "").replace("_", "/")
                frames[symbol] = pd.read_parquet(path)
            symbols = list(frames)
        else:
            n = args.replay_synthetic or 3000
            for symbol in symbols:
                frames[symbol] = make_synthetic_ohlcv(n_bars=n, timeframe=args.timeframe, seed=hash(symbol) % 1000)
        feed = ReplayFeed(frames)
        mode, realtime = "replay", False

    # --- ensemble per symbol ---------------------------------------------
    cost_model = CostModel(
        taker_fee=bt.taker_fee, slippage_base=bt.slippage_base,
        slippage_high_vol=bt.slippage_high_vol, latency_bars=bt.latency_bars,
        fill_ratio=bt.fill_ratio,
    )
    mapper = ActionMapper(
        vol_stop_scale=tc.vol_stop_scale, min_stop_pct=tc.min_stop_pct,
        max_stop_pct=tc.max_stop_pct, reward_risk=tc.reward_risk,
        cost_model=cost_model, cost_margin=tc.cost_margin,
    )
    strategies = {}
    for symbol in symbols:
        members = build_members(args.fresh_agents, config.models_dir)
        if args.fresh_agents:
            # Fit normalization stats on a synthetic frame so untrained
            # agents still see sane inputs.
            seed_df = extract_features(make_synthetic_ohlcv(n_bars=2000, timeframe=args.timeframe))
            for m in members:
                m.stats = FeatureStats.fit(seed_df, m.feature_columns)
        strategies[symbol] = EnsembleStrategy(
            members, mapper, vote_threshold=pc.vote_threshold, online_learning=True,
        )

    # --- guardians + journal ----------------------------------------------
    risk_engine = RiskEngine(RiskLimits(**rc.model_dump()))
    journal = TradeJournal(Path(config.resolve_path("data")) / Path(pc.journal_path).name)
    alerter = TelegramAlerter(
        token=secrets.telegram_bot_token, chat_id=secrets.telegram_chat_id,
        throttle_seconds=pc.alert_throttle_seconds,
    )
    dashboard = DashboardWriter(config.reports_dir / Path(pc.dashboard_path).name, mode=mode)

    trader = PaperTrader(
        symbols=symbols, strategies=strategies, risk_engine=risk_engine,
        cost_model=cost_model,
        journal=journal, alerter=alerter, dashboard=dashboard,
        initial_equity=bt.initial_equity, warmup_bars=pc.warmup_bars,
        buffer_bars=pc.buffer_bars, dashboard_every=pc.dashboard_every,
        regime_window_bars=min(300, pc.buffer_bars // 2), regime_threshold=bt.regime_threshold,
        timeframe_ms=tf_ms, watchdog_stale_multiple=pc.watchdog_stale_multiple,
        realtime=realtime,
    )

    if realtime:
        def watchdog_loop():
            while True:
                time.sleep(30)
                trader.check_feed_health()
        threading.Thread(target=watchdog_loop, daemon=True, name="watchdog").start()

    # The journal persists across runs, so "this session" has to be bounded
    # explicitly. Without this the summary reports every trade the file has
    # ever held: a run that took no trades still printed the previous run's
    # losses next to its own untouched final equity.
    session_start_id = journal.last_trade_id()

    try:
        trader.run(feed, max_bars=args.max_bars)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    trades = journal.trades(after_id=session_start_id)
    print(f"\n=== Paper trading session summary ({mode}) ===")
    print(f"bars processed:  {trader._bars_processed}")
    print(f"closed trades:   {len(trades)}")
    if len(trades):
        print(f"win rate:        {(trades['pnl'] > 0).mean():.1%}")
        print(f"total pnl:       {trades['pnl'].sum():+.2f}")
        print(f"total fees:      {trades['fees'].sum():.2f}")
        print(f"avg R:           {trades['r_multiple'].mean():+.3f}")
    else:
        print("                 (took no trades this session)")
    print(f"final equity:    {trader.equity:,.2f} (from {trader.initial_equity:,.2f})")
    if journal.trade_count() > len(trades):
        print(f"journal total:   {journal.trade_count()} trades incl. earlier sessions")
    print(f"journal:         {journal.path}")
    print(f"dashboard:       {dashboard.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
