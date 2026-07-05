import numpy as np
import pandas as pd
import pytest

from cognition.backtest.costs import CostModel
from cognition.backtest.simulator import EventDrivenBacktester, Signal
from cognition.backtest.strategies import ScriptedStrategy

TF_MS = 60_000


def flat_df(n=20, price=100.0):
    """Every bar open=high=low=close=price. Tests mutate specific bars."""
    return pd.DataFrame(
        {
            "timestamp": [1_700_000_000_000 + i * TF_MS for i in range(n)],
            "open": [price] * n,
            "high": [price] * n,
            "low": [price] * n,
            "close": [price] * n,
            "volume": [10.0] * n,
        }
    )


def make_engine(**overrides):
    cost_kwargs = {k: overrides.pop(k) for k in ("taker_fee", "slippage_base", "slippage_high_vol", "latency_bars", "fill_ratio") if k in overrides}
    cm = CostModel(**cost_kwargs)
    return EventDrivenBacktester(cm, initial_equity=10_000.0, **overrides), cm


def test_signal_fills_at_next_bar_open_with_adverse_slippage():
    df = flat_df()
    engine, cm = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.05)})
    result = engine.run(df, strat)
    assert len(result.trades) == 1
    trade = result.trades[0]
    # Fill at bar 3 (latency 1), not bar 2 — no same-bar lookahead.
    assert trade.entry_timestamp == int(df.loc[3, "timestamp"])
    assert trade.entry_price == pytest.approx(100.0 * 1.001)


def test_latency_of_two_bars_delays_the_fill():
    df = flat_df()
    engine, _ = make_engine(latency_bars=2)
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.05)})
    result = engine.run(df, strat)
    assert result.trades[0].entry_timestamp == int(df.loc[4, "timestamp"])


def test_take_profit_exit_arithmetic_is_internally_consistent():
    df = flat_df()
    df.loc[6, "high"] = 103.0  # crosses the take-profit
    engine, cm = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.01, take_profit_pct=0.02)})
    result = engine.run(df, strat)
    trade = result.trades[0]

    assert trade.exit_reason == "take_profit"
    entry = 100.0 * 1.001
    tp_raw = entry * 1.02
    assert trade.exit_price == pytest.approx(tp_raw * 0.999)
    expected_fees = cm.fee(trade.size * trade.entry_price) + cm.fee(trade.size * trade.exit_price)
    assert trade.fees == pytest.approx(expected_fees)
    expected_pnl = (trade.exit_price - trade.entry_price) * trade.size - trade.fees
    assert trade.pnl == pytest.approx(expected_pnl)
    assert result.final_equity == pytest.approx(10_000.0 + trade.pnl)


def test_stop_loss_exit_loses_about_one_r():
    df = flat_df()
    df.loc[6, "low"] = 95.0  # crashes through the stop
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.01)})
    result = engine.run(df, strat)
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.pnl < 0
    # -1R on price, minus fee and slippage drag; must never be better than -1R.
    assert -1.5 < trade.r_multiple <= -1.0


def test_stop_taken_first_when_bar_hits_both_stop_and_target():
    df = flat_df()
    df.loc[6, "high"] = 110.0
    df.loc[6, "low"] = 90.0
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.01, take_profit_pct=0.02)})
    result = engine.run(df, strat)
    assert result.trades[0].exit_reason == "stop_loss"  # conservative assumption


def test_gap_through_stop_fills_at_the_worse_open():
    df = flat_df()
    df.loc[6, "open"] = 97.0
    df.loc[6, "low"] = 96.0
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.01)})
    result = engine.run(df, strat)
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(97.0 * 0.999)  # open, not the stop price
    assert trade.r_multiple < -1.5  # gap made it worse than -1R


def test_short_side_mirrors_long_side():
    df = flat_df()
    df.loc[6, "low"] = 97.0  # short take-profit region
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=-1, stop_loss_pct=0.01, take_profit_pct=0.02)})
    result = engine.run(df, strat)
    trade = result.trades[0]
    assert trade.direction == -1
    assert trade.entry_price == pytest.approx(100.0 * 0.999)  # short sells lower on entry
    assert trade.exit_reason == "take_profit"
    assert trade.pnl > 0


def test_position_size_risks_exactly_the_configured_fraction():
    df = flat_df()
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.05)})
    result = engine.run(df, strat)
    trade = result.trades[0]
    stop_distance = trade.entry_price * 0.05
    assert trade.size * stop_distance == pytest.approx(10_000.0 * 0.01)  # 1% of equity at risk


def test_position_notional_is_capped_at_max_fraction_of_equity():
    df = flat_df()
    engine, _ = make_engine()
    # Tiny 0.1% stop → risk-based size would be ~10x equity. Must be capped.
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.001)})
    result = engine.run(df, strat)
    trade = result.trades[0]
    assert trade.size * trade.entry_price == pytest.approx(10_000.0 * 0.95)


def test_partial_fill_ratio_scales_down_size():
    df = flat_df()
    engine_full, _ = make_engine()
    engine_half, _ = make_engine(fill_ratio=0.5)
    strat = lambda: ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.05)})  # noqa: E731
    full = engine_full.run(df, strat()).trades[0]
    half = engine_half.run(df, strat()).trades[0]
    assert half.size == pytest.approx(full.size * 0.5)


def test_opposite_signal_closes_position_at_next_open():
    df = flat_df()
    engine, _ = make_engine()
    strat = ScriptedStrategy({
        2: Signal(direction=1, stop_loss_pct=0.05),
        5: Signal(direction=-1, stop_loss_pct=0.05),
    })
    result = engine.run(df, strat)
    trade = result.trades[0]
    assert trade.exit_reason == "signal"
    assert trade.exit_timestamp == int(df.loc[6, "timestamp"])  # signal at 5 + latency 1


def test_open_position_is_force_closed_at_end_of_data():
    df = flat_df(n=10)
    engine, _ = make_engine()
    strat = ScriptedStrategy({6: Signal(direction=1, stop_loss_pct=0.05)})
    result = engine.run(df, strat)
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_data"
    assert result.equity_curve.iloc[-1] == pytest.approx(result.final_equity)


def test_max_holding_bars_forces_a_time_exit():
    df = flat_df(n=20)
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.05, max_holding_bars=4)})
    result = engine.run(df, strat)
    trade = result.trades[0]
    assert trade.exit_reason == "time"
    assert trade.bars_held == 4


def test_no_trades_before_start_index():
    df = flat_df()
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.05)})
    result = engine.run(df, strat, start_index=5)
    assert result.trades == []


def test_signals_without_a_stop_loss_are_refused():
    df = flat_df()
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.0)})
    result = engine.run(df, strat)
    assert result.trades == []


def test_risk_per_trade_above_one_percent_is_rejected():
    with pytest.raises(ValueError):
        EventDrivenBacktester(CostModel(), risk_per_trade=0.02)


def test_high_volatility_regime_widens_entry_slippage():
    df = flat_df()
    df["volatility_regime"] = "high"
    engine, _ = make_engine()
    strat = ScriptedStrategy({2: Signal(direction=1, stop_loss_pct=0.05)})
    result = engine.run(df, strat)
    assert result.trades[0].entry_price == pytest.approx(100.0 * 1.005)


def test_final_equity_equals_initial_plus_sum_of_trade_pnl():
    rng = np.random.default_rng(3)
    n = 500
    prices = 100 + np.cumsum(rng.normal(0, 0.3, n))
    df = flat_df(n)
    df["open"] = prices
    df["close"] = prices
    df["high"] = prices + 0.5
    df["low"] = prices - 0.5
    engine, _ = make_engine()
    signals = {i: Signal(direction=1 if (i // 40) % 2 == 0 else -1, stop_loss_pct=0.005, take_profit_pct=0.01) for i in range(5, n - 5, 20)}
    result = engine.run(df, ScriptedStrategy(signals))
    assert len(result.trades) > 5
    assert result.final_equity == pytest.approx(result.initial_equity + sum(t.pnl for t in result.trades))
