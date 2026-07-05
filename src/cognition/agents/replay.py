"""Ring-buffer experience replay."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._next = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(self, obs, action, reward, next_obs, done) -> None:
        idx = self._next
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        idx = self._rng.integers(0, self._size, size=batch_size)
        return (
            self.obs[idx], self.actions[idx], self.rewards[idx],
            self.next_obs[idx], self.dones[idx],
        )
