from cognition.learning.journal import TradeJournal


def make_journal(tmp_path):
    return TradeJournal(tmp_path / "journal.db")


def trade_kwargs(**overrides):
    base = dict(
        symbol="BTC/USDT", direction=1, entry_ts=1000, exit_ts=2000,
        entry_price=50_000.0, exit_price=50_200.0, size=0.1,
        pnl=15.0, r_multiple=0.75, fees=5.0, slippage=2.0,
        exit_reason="take_profit", bars_held=12, mae_r=0.4,
        market_regime="bull", session="us",
        agent_votes={"momentum": {"direction": 1, "confidence": 0.8}},
        feature_snapshot={"rsi_14": 61.2, "volatility": 0.001},
        equity_after=10_015.0,
    )
    base.update(overrides)
    return base


def test_trade_roundtrip_preserves_all_spec_fields(tmp_path):
    journal = make_journal(tmp_path)
    journal.record_trade(**trade_kwargs())
    df = journal.trades()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["symbol"] == "BTC/USDT"
    assert row["pnl"] == 15.0
    assert row["r_multiple"] == 0.75
    assert row["mae_r"] == 0.4
    assert row["market_regime"] == "bull" and row["session"] == "us"
    assert row["agent_votes"]["momentum"]["confidence"] == 0.8
    assert row["feature_snapshot"]["rsi_14"] == 61.2
    assert row["equity_after"] == 10_015.0


def test_journal_persists_across_reopen(tmp_path):
    journal = make_journal(tmp_path)
    journal.record_trade(**trade_kwargs())
    journal.close()
    reopened = make_journal(tmp_path)
    assert reopened.trade_count() == 1


def test_equity_curve_and_events(tmp_path):
    journal = make_journal(tmp_path)
    for k in range(5):
        journal.record_equity(1000 + k, 10_000.0 + k)
    journal.record_event(1004, "critical", "circuit_breaker", "halted at -5%")
    curve = journal.equity_curve()
    assert len(curve) == 5
    assert curve["equity"].iloc[-1] == 10_004.0
    events = journal.events()
    assert events.iloc[0]["kind"] == "circuit_breaker"


def test_trades_limit_returns_most_recent(tmp_path):
    journal = make_journal(tmp_path)
    for k in range(10):
        journal.record_trade(**trade_kwargs(exit_ts=2000 + k, pnl=float(k)))
    recent = journal.trades(limit=3)
    assert len(recent) == 3
    assert set(recent["pnl"]) == {9.0, 8.0, 7.0}


# --- session scoping ----------------------------------------------------
#
# The journal is durable across runs by design. A caller summarising "this
# session" therefore has to bound it, or it reports the whole file as if it
# had just happened -- which is exactly what the paper trader did: a run that
# took no trades printed the previous run's losses beside its own untouched
# equity.


def test_last_trade_id_is_zero_on_empty_journal(tmp_path):
    assert make_journal(tmp_path).last_trade_id() == 0


def test_trades_after_id_scopes_to_one_session(tmp_path):
    journal = make_journal(tmp_path)
    journal.record_trade(**trade_kwargs(exit_ts=2000, pnl=-50.0))
    journal.record_trade(**trade_kwargs(exit_ts=2001, pnl=-25.0))
    boundary = journal.last_trade_id()
    journal.record_trade(**trade_kwargs(exit_ts=2002, pnl=10.0))

    assert len(journal.trades()) == 3                 # whole file
    session = journal.trades(after_id=boundary)
    assert len(session) == 1
    assert session["pnl"].sum() == 10.0


def test_a_session_that_traded_nothing_reports_nothing(tmp_path):
    """The regression: zero trades this run must not surface earlier losses."""
    journal = make_journal(tmp_path)
    journal.record_trade(**trade_kwargs(pnl=-926.59))
    boundary = journal.last_trade_id()

    assert journal.trades(after_id=boundary).empty
    assert journal.trade_count() == 1   # still durable, just not ours


def test_last_trade_id_tracks_inserts(tmp_path):
    journal = make_journal(tmp_path)
    assert journal.last_trade_id() == 0
    journal.record_trade(**trade_kwargs())
    first = journal.last_trade_id()
    assert first > 0
    journal.record_trade(**trade_kwargs(exit_ts=3000))
    assert journal.last_trade_id() > first
