"""Dueling Q-network: the spec's "DQN with actor-critic components".

The shared trunk feeds two heads:
- a state-value head (the critic): V(s)
- an advantage head over actions (the actor stream): A(s, a)

Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.trunk(obs)
        value = self.value_head(z)
        advantage = self.advantage_head(z)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)
