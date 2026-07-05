"""Tests for metrics, regime labeling, walk-forward/lockbox splits, and
Monte Carlo perturbation."""
import numpy as np
import pandas as pd
import pytest

from cognition.backtest.costs import CostModel
from cognition.backtest.metrics import compute_metrics, max_drawdown, sharpe_ratio, stats_by_group, trades_to_frame
from cognition.backtest.montecarlo import perturb_ohlcv, run_monte_carlo
from cognition.backtest.regimes import label_market_regimes
from cognition.backtest.simulator import EventDrivenBacktester, Signal, Trade
from cognition.backtest.strategies import ScriptedStrategy, SmaCrossoverStrategy
from cognition.backtest.synthetic import make_synthetic_ohlcv
from cognition.backtest.walkforward import run_walk_forward, split_lockbox, walk_forward_splits
from cognition.utils.timeframes import MS_PER_DAY, bars_per_year, timeframe_to_ms


def make_trade(pnl, r=None, regime="bull"):
    return Trade(
        direction=1, entry_timestamp=0, exit_timestamp=1, entry_price=100.0,
        exit_price=100.0 + pnl, size=1.0, pnl=pnl, r_multiple=r if r is not None else pnl / 10,
        fees=0.1, slippage_cost=0.05, exit_reason="take_profit", bars_held=5,
        mae_r=0.2, market_regime=regime, session="us",
    )


# ---------- metrics ----------

def test_max_drawdown_on_known_curve():
    curve = pd.Series([100.0, 110.0, 99.0, 105.0, 120.0])
    assert max_drawdown(curve) == pytest.approx(0.1)  # 110 -> 99


def test_sharpe_positive_for_rising_curve():
    curve = pd.Series(np.linspace(100, 200, 100) + np.random.default_rng(0).normal(0, 0.1, 100))
    assert sharpe_ratio(curve, bars_per_year("1m")) > 0


def test_trade_stats_win_rate_and_profit_factor():
    trades = [make_trade(10), make_trade(10), make_trade(-5)]
    df = trades_to_frame(trades)
    from cognition.backtest.metrics import trade_stats
    stats = trade_stats(df)
    assert stats["num_trades"] == 3
    assert stats["win_rate"] == pytest.approx(2 / 3)
    assert stats["profit_factor"] == pytest.approx(20 / 5)


def test_stats_by_group_splits_regimes():
    trades = [make_trade(10, regime="bull"), make_trade(-5, regime="bear"), make_trade(3, regime="bear")]
    grouped = stats_by_group(trades_to_frame(trades), "market_regime")
    assert set(grouped.index) == {"bull", "bear"}
    assert grouped.loc["bear", "num_trades"] == 2


# ---------- regimes ----------

def test_regime_labels_match_synthetic_segments():
    n = 4000
    ts = [1_700_000_000_000 + i * 60_000 for i in range(n)]
    up = 100 * np.cumprod(1 + np.full(n // 2, 0.0002))
    flat = np.full(n - n // 2, up[-1])
    close = np.concatenate([up, flat])
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    labeled = label_market_regimes(df, window_bars=500, threshold=0.03)
    assert (labeled["market_regime"].iloc[:500] == "unknown").all()
    assert labeled["market_regime"].iloc[1500] == "bull"     # mid-uptrend
    assert labeled["market_regime"].iloc[-1] == "sideways"   # flat tail
    # No lookahead: labeling twice on a truncated frame gives identical labels.
    relabeled = label_market_regimes(df.iloc[:2000].reset_index(drop=True), window_bars=500, threshold=0.03)
    assert (relabeled["market_regime"] == labeled["market_regime"].iloc[:2000].to_numpy()).all()


# ---------- walk-forward & lockbox ----------

def test_lockbox_excludes_recent_months():
    n = 10_000
    df = pd.DataFrame({
        "timestamp": [1_600_000_000_000 + i * MS_PER_DAY for i in range(n)],  # daily spacing
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
    })
    main, lockbox = split_lockbox(df, months=6)
    assert len(lockbox) == 6 * 30 + 1
    assert main["timestamp"].max() < lockbox["timestamp"].min()
    assert len(main) + len(lockbox) == n


def test_walk_forward_test_windows_never_overlap_and_train_precedes_test():
    df = make_synthetic_ohlcv(n_bars=5000, seed=1)
    prev_test_end = None
    for train, test, start_index in walk_forward_splits(df, train_bars=1000, test_bars=500, warmup_bars=100):
        assert len(train) == 1000
        assert len(test) == 600  # 100 warmup + 500 test
        assert start_index == 100
        test_start_ts = test["timestamp"].iloc[start_index]
        assert train["timestamp"].iloc[-1] < test_start_ts
        if prev_test_end is not None:
            assert test_start_ts > prev_test_end
        prev_test_end = test["timestamp"].iloc[-1]


def test_run_walk_forward_produces_window_metrics():
    df = make_synthetic_ohlcv(n_bars=6000, seed=2)
    engine = EventDrivenBacktester(CostModel(), initial_equity=10_000)
    windows, trades = run_walk_forward(
        df, engine, SmaCrossoverStrategy,
        train_bars=1500, test_bars=1000, warmup_bars=100, bars_per_year=bars_per_year("1m"),
    )
    assert len(windows) == 4  # (6000 - 1500) // 1000
    for w in windows:
        assert "sharpe" in w.metrics and "num_trades" in w.metrics


# ---------- Monte Carlo ----------

def test_perturbed_path_differs_but_stays_coherent():
    df = make_synthetic_ohlcv(n_bars=2000, seed=3)
    rng = np.random.default_rng(0)
    perturbed = perturb_ohlcv(df, rng, noise_scale=0.25)
    assert not np.allclose(perturbed["close"], df["close"])
    assert perturbed["close"].iloc[0] == pytest.approx(df["close"].iloc[0])  # anchored start
    assert (perturbed["high"] >= perturbed["low"]).all()
    assert (perturbed["close"] > 0).all()
    assert (perturbed["timestamp"] == df["timestamp"]).all()


def test_monte_carlo_is_reproducible_with_seed():
    df = make_synthetic_ohlcv(n_bars=1500, seed=4)
    engine = EventDrivenBacktester(CostModel(), initial_equity=10_000)
    a = run_monte_carlo(df, engine, SmaCrossoverStrategy, runs=3, bars_per_year=bars_per_year("1m"), seed=11)
    b = run_monte_carlo(df, engine, SmaCrossoverStrategy, runs=3, bars_per_year=bars_per_year("1m"), seed=11)
    assert len(a) == 3
    pd.testing.assert_frame_equal(a, b)


# ---------- synthetic + full-stack smoke ----------

def test_synthetic_data_is_valid_ohlcv():
    df = make_synthetic_ohlcv(n_bars=3000, timeframe="5m")
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["timestamp"].diff().dropna() == timeframe_to_ms("5m")).all()


def test_full_stack_smoke_sma_on_synthetic():
    df = make_synthetic_ohlcv(n_bars=4000, seed=5)
    df = label_market_regimes(df, window_bars=400)
    engine = EventDrivenBacktester(CostModel(), initial_equity=10_000)
    result = engine.run(df, SmaCrossoverStrategy())
    metrics = compute_metrics(result, bars_per_year("1m"))
    assert metrics["num_trades"] > 10
    assert metrics["total_fees"] > 0       # costs are actually being charged
    assert metrics["total_slippage"] > 0
    regime_split = stats_by_group(trades_to_frame(result.trades), "market_regime")
    assert not regime_split.empty
