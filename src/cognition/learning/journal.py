"""Trade Journal — the system's memory (Layer 3).

SQLite (stdlib, upgradeable to PostgreSQL behind this same interface).
Every closed trade records everything the spec demands: entry/exit,
size, direction, duration, per-agent votes + confidences, the full
feature snapshot at entry, regime tag, session, P&L after real
fees/slippage, R-multiple, and max adverse excursion. Also keeps an
equity curve and an operational event log for Guardian B.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry_ts INTEGER NOT NULL,
    exit_ts INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    size REAL NOT NULL,
    pnl REAL NOT NULL,
    r_multiple REAL NOT NULL,
    fees REAL NOT NULL,
    slippage REAL NOT NULL,
    exit_reason TEXT,
    bars_held INTEGER,
    mae_r REAL,
    market_regime TEXT,
    session TEXT,
    agent_votes TEXT,
    feature_snapshot TEXT,
    equity_after REAL
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts INTEGER NOT NULL,
    equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    ts INTEGER NOT NULL,
    level TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_exit_ts ON trades (exit_ts);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve (ts);
"""

TRADE_COLUMNS = [
    "symbol", "direction", "entry_ts", "exit_ts", "entry_price", "exit_price",
    "size", "pnl", "r_multiple", "fees", "slippage", "exit_reason", "bars_held",
    "mae_r", "market_regime", "session", "agent_votes", "feature_snapshot",
    "equity_after",
]


class TradeJournal:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- writes -------------------------------------------------------

    def record_trade(
        self,
        *,
        symbol: str,
        direction: int,
        entry_ts: int,
        exit_ts: int,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        r_multiple: float,
        fees: float,
        slippage: float,
        exit_reason: str,
        bars_held: int,
        mae_r: float,
        market_regime: object = None,
        session: object = None,
        agent_votes: dict | None = None,
        feature_snapshot: dict | None = None,
        equity_after: float = 0.0,
    ) -> int:
        row = (
            symbol, direction, entry_ts, exit_ts, entry_price, exit_price,
            size, pnl, r_multiple, fees, slippage, exit_reason, bars_held,
            mae_r,
            str(market_regime) if market_regime is not None else None,
            str(session) if session is not None else None,
            json.dumps(agent_votes, default=str) if agent_votes is not None else None,
            json.dumps(feature_snapshot, default=str) if feature_snapshot is not None else None,
            equity_after,
        )
        placeholders = ", ".join("?" * len(TRADE_COLUMNS))
        cursor = self._conn.execute(
            f"INSERT INTO trades ({', '.join(TRADE_COLUMNS)}) VALUES ({placeholders})", row
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def record_equity(self, ts: int, equity: float) -> None:
        self._conn.execute("INSERT INTO equity_curve (ts, equity) VALUES (?, ?)", (ts, equity))
        self._conn.commit()

    def record_event(self, ts: int, level: str, kind: str, message: str) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, level, kind, message) VALUES (?, ?, ?, ?)",
            (ts, level, kind, message),
        )
        self._conn.commit()

    # ---- reads --------------------------------------------------------

    def trade_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])

    def trades(self, limit: int | None = None) -> pd.DataFrame:
        query = "SELECT * FROM trades ORDER BY exit_ts"
        if limit is not None:
            query += f" DESC LIMIT {int(limit)}"
        df = pd.read_sql_query(query, self._conn)
        for col in ("agent_votes", "feature_snapshot"):
            if col in df.columns:
                df[col] = df[col].map(lambda v: json.loads(v) if v else None)
        return df

    def equity_curve(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM equity_curve ORDER BY ts", self._conn)

    def events(self, limit: int = 100) -> pd.DataFrame:
        return pd.read_sql_query(
            f"SELECT * FROM events ORDER BY ts DESC LIMIT {int(limit)}", self._conn
        )
