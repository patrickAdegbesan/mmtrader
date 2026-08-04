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


# --- liquidity role: maker vs taker -------------------------------------
#
# The executor quotes maker-first, so the cost model has to be able to price
# that. These pin the distinction, including the part that makes maker fills
# expensive rather than free.


def test_maker_fee_defaults_to_taker_rate():
    """Spot-like venues charge both sides the same; absent an explicit maker
    rate the model must not invent a discount."""
    cm = CostModel(taker_fee=0.001)
    assert cm.fee_rate("maker") == 0.001
    assert cm.fee_rate("taker") == 0.001


def test_maker_fee_used_when_given():
    cm = CostModel(taker_fee=0.00055, maker_fee=0.0002)  # Bybit perp VIP0
    assert cm.fee_rate("maker") == 0.0002
    assert cm.fee_rate("taker") == 0.00055
    assert cm.fee(10_000, "maker") == pytest.approx(2.0)
    assert cm.fee(10_000, "taker") == pytest.approx(5.5)


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        CostModel().fee_rate("hedger")


def test_maker_pays_adverse_selection_not_spread():
    """A maker set the price, so it never crosses the spread -- but it is
    adversely selected instead. Regime slippage must not apply to it."""
    cm = CostModel(slippage_base=0.001, slippage_high_vol=0.005,
                   maker_adverse_selection=0.0003)
    assert cm.slippage_rate("high", "maker") == 0.0003   # regime irrelevant
    assert cm.slippage_rate("low", "maker") == 0.0003
    assert cm.slippage_rate("high", "taker") == 0.005


def test_maker_default_adverse_selection_is_zero_and_that_is_optimistic():
    """Documented default: a maker that always fills and is never picked off.
    Deliberately optimistic, and the reason maker backtests need a measured
    number rather than this one."""
    cm = CostModel()
    assert cm.slippage_rate("high", "maker") == 0.0


def test_maker_round_trip_is_cheaper_than_taker():
    cm = CostModel(taker_fee=0.00055, maker_fee=0.0002, slippage_base=0.001)
    taker = cm.round_trip_cost("low", "taker")
    maker = cm.round_trip_cost("low", "maker")
    assert taker == pytest.approx(2 * 0.00055 + 2 * 0.001)   # 0.311%
    assert maker == pytest.approx(2 * 0.0002)                # 0.04%
    assert maker < taker


def test_maker_entry_and_exit_remain_adverse():
    """Adverse selection still moves the fill against the trader, in both
    directions and on both legs -- it is a cost, not a rebate."""
    cm = CostModel(maker_adverse_selection=0.001)
    assert cm.entry_price(100.0, 1, "low", "maker") == pytest.approx(100.1)
    assert cm.entry_price(100.0, -1, "low", "maker") == pytest.approx(99.9)
    assert cm.exit_price(100.0, 1, "low", "maker") == pytest.approx(99.9)
    assert cm.exit_price(100.0, -1, "low", "maker") == pytest.approx(100.1)


def test_effective_cost_blends_maker_and_taker_by_fill_probability():
    """The executor crosses for whatever does not fill, so realised cost sits
    between the two rates."""
    cm = CostModel(taker_fee=0.00055, maker_fee=0.0002, slippage_base=0.001,
                   maker_fill_probability=0.5)
    maker_rt = 2 * 0.0002                 # 0.04%
    taker_rt = 2 * 0.00055 + 2 * 0.001    # 0.31%
    assert cm.effective_round_trip_cost("low") == pytest.approx(
        0.5 * maker_rt + 0.5 * taker_rt
    )


def test_effective_cost_equals_maker_cost_when_always_filled():
    cm = CostModel(taker_fee=0.00055, maker_fee=0.0002, maker_fill_probability=1.0)
    assert cm.effective_round_trip_cost("low") == pytest.approx(
        cm.round_trip_cost("low", "maker")
    )


def test_taker_defaults_unchanged():
    """Existing callers pass no role and must keep the old behaviour."""
    cm = CostModel(taker_fee=0.001, slippage_base=0.001)
    assert cm.round_trip_cost("low") == pytest.approx(0.004)
    assert cm.fee(10_000) == pytest.approx(10.0)
    assert cm.entry_price(100.0, 1, "low") == pytest.approx(100.1)


def test_invalid_maker_params_rejected():
    with pytest.raises(ValueError):
        CostModel(maker_fee=-0.0001)
    with pytest.raises(ValueError):
        CostModel(maker_fill_probability=0.0)
    with pytest.raises(ValueError):
        CostModel(maker_fill_probability=1.5)
    with pytest.raises(ValueError):
        CostModel(maker_adverse_selection=-0.001)
