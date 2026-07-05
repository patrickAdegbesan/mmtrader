"""Per-agent feature specifications for the 4-agent ensemble.

Each agent is an independent DQN specialized for one market behavior by
seeing only the features relevant to it (same data, different lens).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    feature_columns: list[str] = field(default_factory=list)


AGENT_SPECS: dict[str, AgentSpec] = {
    "momentum": AgentSpec(
        name="momentum",
        description="Price velocity and acceleration; thrives in trending bursts.",
        feature_columns=[
            "price_velocity", "price_acceleration", "rsi_14", "macd_diff",
            "bollinger_position", "volume_delta", "volatility",
        ],
    ),
    "mean_reversion": AgentSpec(
        name="mean_reversion",
        description="Oversold/overbought snapbacks; thrives in ranging markets.",
        feature_columns=[
            "rsi_14", "bollinger_position", "sma21_distance", "price_velocity",
            "price_acceleration", "volatility",
        ],
    ),
    "volume_breakout": AgentSpec(
        name="volume_breakout",
        description="Volume surges and range breaks.",
        feature_columns=[
            "volume_zscore", "volume_delta", "donchian_position",
            "bollinger_position", "price_velocity", "volatility",
        ],
    ),
    "microstructure": AgentSpec(
        name="microstructure",
        description=(
            "Order-book imbalance, spread behavior, liquidity shifts. "
            "Currently trained on candle-derived PROXIES (range_pct, "
            "close_in_range); real order-book features replace these when "
            "the live WebSocket feed lands in Milestone 6."
        ),
        feature_columns=[
            "range_pct", "close_in_range", "volume_zscore", "price_velocity",
            "volatility",
        ],
    ),
}

AGENT_NAMES = list(AGENT_SPECS)
