"""Guardian A tests — every non-negotiable rule, including the config
validators that refuse to loosen them."""
import pytest
from pydantic import ValidationError

from cognition.config import RiskConfig
from cognition.risk.engine import RiskEngine, RiskLimits, TradeProposal

T0 = 1_700_006_400_000  # exactly UTC midnight, so intraday offsets stay in one day
HOUR = 3_600_000
DAY = 24 * HOUR


def proposal(ts=T0, symbol="BTC/USDT", direction=1, price=50_000.0,
             stop=0.004, tp=0.006, win_prob=None, rr=None):
    return TradeProposal(
        symbol=symbol, direction=direction, price=price,
        stop_loss_pct=stop, take_profit_pct=tp,
        timestamp_ms=ts, win_prob=win_prob, reward_risk=rr,
    )


def engine(**overrides):
    return RiskEngine(RiskLimits(**overrides))


# ---------- rule 1: 1% cap + fractional Kelly ----------
# Sizing tests use a high exposure ceiling so the exposure cap (rule 5,
# tested separately) doesn't mask the risk-sizing behavior under test:
# with a 0.4% stop, 1% risk implies 2.5x equity notional.

def test_approved_trade_risks_at_most_one_percent():
    p = proposal()
    d = engine(max_total_exposure=5.0).evaluate(p, equity=10_000.0)
    assert d.approved
    assert d.risk_amount <= 100.0 * 1.0001
    assert d.quantity * p.price * d.stop_loss_pct == pytest.approx(d.risk_amount)


def test_kelly_reduces_risk_below_cap_when_edge_is_small():
    # Strong edge: kelly = 0.55 - 0.45/1.5 = 0.25, half-kelly 12.5% -> cap binds at 1%.
    d_strong = engine(max_total_exposure=5.0).evaluate(proposal(win_prob=0.55, rr=1.5), equity=10_000.0)
    assert d_strong.risk_fraction == pytest.approx(0.01, rel=1e-3)
    # Tiny edge: kelly = 0.505 - 0.495/1.0 = 0.01, half-kelly 0.5% -> BELOW the cap.
    d_weak = engine(max_total_exposure=5.0).evaluate(proposal(win_prob=0.505, rr=1.0), equity=10_000.0)
    assert d_weak.approved
    assert d_weak.risk_fraction == pytest.approx(0.005, rel=1e-2)
    assert d_weak.risk_fraction < 0.01


def test_kelly_never_exceeds_the_cap_even_with_huge_edge():
    d = engine(max_total_exposure=5.0).evaluate(proposal(win_prob=0.95, rr=3.0), equity=10_000.0)
    assert d.risk_fraction <= 0.01 * 1.0001


def test_no_edge_proposal_is_rejected():
    d = engine().evaluate(proposal(win_prob=0.4, rr=1.0), equity=10_000.0)
    assert not d.approved
    assert "no edge" in d.reason


# ---------- rule 2: daily circuit breaker ----------

def test_circuit_breaker_trips_at_minus_five_percent():
    e = engine()
    assert e.evaluate(proposal(ts=T0), equity=10_000.0).approved       # anchors the day
    d = e.evaluate(proposal(ts=T0 + HOUR), equity=9_499.0)             # -5.01%
    assert not d.approved
    assert "circuit breaker" in d.reason
    assert e.is_halted


def test_halt_persists_for_rest_of_day_even_after_recovery():
    e = engine()
    e.evaluate(proposal(ts=T0), equity=10_000.0)
    e.evaluate(proposal(ts=T0 + HOUR), equity=9_400.0)                 # trips
    d = e.evaluate(proposal(ts=T0 + 2 * HOUR), equity=9_900.0)         # recovered, still halted
    assert not d.approved
    assert "review mode" in d.reason


def test_halt_resets_on_next_utc_day():
    e = engine()
    e.evaluate(proposal(ts=T0), equity=10_000.0)
    e.evaluate(proposal(ts=T0 + HOUR), equity=9_400.0)                 # trips
    d = e.evaluate(proposal(ts=T0 + DAY), equity=9_400.0)              # new day, new anchor
    assert d.approved


def test_four_percent_loss_does_not_trip():
    e = engine()
    e.evaluate(proposal(ts=T0), equity=10_000.0)
    d = e.evaluate(proposal(ts=T0 + HOUR), equity=9_600.0)
    assert d.approved


# ---------- rule 3: drawdown governor ----------

def test_drawdown_governor_shrinks_size_beyond_threshold():
    e = engine(max_total_exposure=5.0)
    e.evaluate(proposal(ts=T0), equity=10_000.0)                       # peak = 10k
    d = e.evaluate(proposal(ts=T0 + DAY), equity=8_500.0)              # 15% dd (new day: no breaker)
    assert d.approved
    assert d.risk_fraction == pytest.approx(0.005, rel=1e-3)           # 1% * 0.5 governor
    assert any("drawdown governor" in a for a in d.adjustments)


def test_no_governor_below_threshold():
    e = engine(max_total_exposure=5.0)
    e.evaluate(proposal(ts=T0), equity=10_000.0)
    d = e.evaluate(proposal(ts=T0 + DAY), equity=9_500.0)              # 5% dd
    assert d.risk_fraction == pytest.approx(0.01, rel=1e-3)


# ---------- rule 4: overtrading guard ----------

def test_hourly_trade_limit_enforced():
    e = engine(max_trades_per_hour=2, max_trades_per_day=100)
    for k in range(2):
        d = e.evaluate(proposal(ts=T0 + k * 60_000, symbol=f"S{k}"), equity=10_000.0)
        assert d.approved
        e.record_open(f"S{k}", d.notional, T0 + k * 60_000)
        e.record_close(f"S{k}", 10_000.0)
    d = e.evaluate(proposal(ts=T0 + 3 * 60_000, symbol="S9"), equity=10_000.0)
    assert not d.approved and "hour" in d.reason
    # The window slides: an hour later it's allowed again.
    d2 = e.evaluate(proposal(ts=T0 + HOUR + 4 * 60_000, symbol="S9"), equity=10_000.0)
    assert d2.approved


def test_daily_trade_limit_enforced():
    e = engine(max_trades_per_hour=100, max_trades_per_day=3)
    for k in range(3):
        ts = T0 + k * 2 * HOUR
        d = e.evaluate(proposal(ts=ts, symbol=f"S{k}"), equity=10_000.0)
        assert d.approved
        e.record_open(f"S{k}", d.notional, ts)
        e.record_close(f"S{k}", 10_000.0)
    d = e.evaluate(proposal(ts=T0 + 7 * HOUR, symbol="S9"), equity=10_000.0)
    assert not d.approved and "today" in d.reason


# ---------- rule 5: exposure limits ----------

def test_concurrent_position_limit():
    e = engine(max_concurrent_positions=1)
    d1 = e.evaluate(proposal(symbol="BTC/USDT"), equity=10_000.0)
    assert d1.approved
    e.record_open("BTC/USDT", d1.notional, T0)
    d2 = e.evaluate(proposal(ts=T0 + 60_000, symbol="ETH/USDT", price=3_000.0), equity=10_000.0)
    assert not d2.approved and "concurrent" in d2.reason
    e.record_close("BTC/USDT", 10_000.0)
    d3 = e.evaluate(proposal(ts=T0 + 120_000, symbol="ETH/USDT", price=3_000.0), equity=10_000.0)
    assert d3.approved


def test_duplicate_symbol_rejected():
    e = engine()
    d1 = e.evaluate(proposal(), equity=10_000.0)
    e.record_open("BTC/USDT", d1.notional, T0)
    d2 = e.evaluate(proposal(ts=T0 + 60_000), equity=10_000.0)
    assert not d2.approved and "already open" in d2.reason


def test_total_exposure_cap_shrinks_notional():
    e = engine(max_total_exposure=0.5, max_concurrent_positions=5)
    # 0.4% stop -> risk sizing wants 1%/0.4% = 2.5x equity notional; cap: 0.5x.
    d = e.evaluate(proposal(), equity=10_000.0)
    assert d.approved
    assert d.notional == pytest.approx(5_000.0)
    assert any("exposure cap" in a for a in d.adjustments)
    assert d.risk_amount < 100.0  # shrunk notional means less risk, never more


def test_total_exposure_full_rejects_new_trades():
    e = engine(max_total_exposure=0.5, max_concurrent_positions=5)
    e.record_open("ETH/USDT", 5_000.0, T0)
    d = e.evaluate(proposal(ts=T0 + 60_000), equity=10_000.0)
    assert not d.approved and "exposure" in d.reason


# ---------- rule 6: SL/TP clamping + structural ----------

def test_reckless_stop_is_clamped():
    d = engine().evaluate(proposal(stop=0.10, tp=0.20), equity=10_000.0)
    assert d.approved
    assert d.stop_loss_pct == 0.02
    assert d.take_profit_pct == 0.05
    assert len(d.adjustments) == 2


def test_microscopic_stop_is_clamped_up():
    d = engine().evaluate(proposal(stop=0.0001), equity=10_000.0)
    assert d.approved
    assert d.stop_loss_pct == 0.001


def test_missing_stop_rejected():
    d = engine().evaluate(proposal(stop=0.0), equity=10_000.0)
    assert not d.approved and "no stop-loss" in d.reason


def test_invalid_direction_rejected():
    d = engine().evaluate(proposal(direction=0), equity=10_000.0)
    assert not d.approved


# ---------- config validators: non-negotiables can't be loosened ----------

def test_config_rejects_risk_above_one_percent():
    with pytest.raises(ValidationError):
        RiskConfig(risk_per_trade=0.02)


def test_config_rejects_daily_halt_above_five_percent():
    with pytest.raises(ValidationError):
        RiskConfig(daily_loss_halt=0.10)


def test_config_rejects_governor_threshold_above_ten_percent():
    with pytest.raises(ValidationError):
        RiskConfig(drawdown_governor_threshold=0.20)


def test_config_rejects_non_shrinking_governor():
    with pytest.raises(ValidationError):
        RiskConfig(drawdown_governor_scale=1.5)


def test_config_accepts_tighter_than_required_values():
    cfg = RiskConfig(risk_per_trade=0.005, daily_loss_halt=0.03)
    assert cfg.risk_per_trade == 0.005
