"""Double-DQN agent over the dueling (value/advantage) network."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from cognition.agents.networks import DuelingQNetwork
from cognition.agents.replay import ReplayBuffer


@dataclass
class DQNConfig:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 500
    target_sync_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    grad_clip: float = 5.0
    seed: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DQNAgent:
    observation_dim: int
    n_actions: int
    config: DQNConfig = field(default_factory=DQNConfig)

    def __post_init__(self):
        torch.manual_seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)
        self.online = DuelingQNetwork(self.observation_dim, self.n_actions, self.config.hidden)
        self.target = DuelingQNetwork(self.observation_dim, self.n_actions, self.config.hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_capacity, self.observation_dim, seed=self.config.seed)
        self.steps = 0

    @property
    def epsilon(self) -> float:
        cfg = self.config
        frac = min(self.steps / max(cfg.epsilon_decay_steps, 1), 1.0)
        return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * frac

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        with torch.no_grad():
            q = self.online(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0))
        return int(q.argmax(dim=-1).item())

    def observe(self, obs, action, reward, next_obs, done) -> float | None:
        """Store a transition and (past warmup) run one learning step.
        Returns the TD loss when a step happened, else None.
        """
        self.buffer.push(obs, action, reward, next_obs, done)
        self.steps += 1
        if len(self.buffer) < max(self.config.warmup_steps, self.config.batch_size):
            return None
        loss = self._learn()
        if self.steps % self.config.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss

    def _learn(self) -> float:
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.config.batch_size)
        obs_t = torch.from_numpy(obs)
        actions_t = torch.from_numpy(actions)
        rewards_t = torch.from_numpy(rewards)
        next_obs_t = torch.from_numpy(next_obs)
        dones_t = torch.from_numpy(dones)

        q = self.online(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online net picks the action, target net values it.
            next_actions = self.online(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.config.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "config": self.config.to_dict(),
            "observation_dim": self.observation_dim,
            "n_actions": self.n_actions,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = state["steps"]
