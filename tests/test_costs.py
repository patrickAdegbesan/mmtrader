import pytest

from cognition.backtest.costs import CostModel


def test_fee_is_proportional_to_notional():
    cm = CostModel(taker_fee=0.001)
    assert cm.fee(10_000) == pytest.approx(10.0)
    assert cm.fee(-10_000) == pytest.approx(10.0)  # fees are never negative


def test_slippage_rate_selects_by_volatility_regime():
    cm = CostModel(slippage_base=0.001, slippage_high_vol=0.005)
    assert cm.slippage_rate("low") == 0.001
    assert cm.slippage_rate("medium") == 0.001
    assert cm.slippage_rate("high") == 0.005
    assert cm.slippage_rate(None) == 0.001  # unknown regime falls back to base


def test_entry_slippage_is_adverse_for_both_directions():
    cm = CostModel(slippage_base=0.001)
    assert cm.entry_price(100.0, 1, "low") == pytest.approx(100.1)   # long buys higher
    assert cm.entry_price(100.0, -1, "low") == pytest.approx(99.9)   # short sells lower


def test_exit_slippage_is_adverse_for_both_directions():
    cm = CostModel(slippage_base=0.001)
    assert cm.exit_price(100.0, 1, "low") == pytest.approx(99.9)     # long sells lower
    assert cm.exit_price(100.0, -1, "low") == pytest.approx(100.1)   # short buys back higher


def test_high_vol_slippage_widens_fills():
    cm = CostModel(slippage_base=0.001, slippage_high_vol=0.005)
    assert cm.entry_price(100.0, 1, "high") == pytest.approx(100.5)


def test_same_bar_execution_is_rejected():
    with pytest.raises(ValueError):
        CostModel(latency_bars=0)


def test_invalid_fill_ratio_rejected():
    with pytest.raises(ValueError):
        CostModel(fill_ratio=0.0)
    with pytest.raises(ValueError):
        CostModel(fill_ratio=1.5)
