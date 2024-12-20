from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from sbx import DQN, PPO, SAC, TD3, TQC


@dataclass
class DummyEnv(gym.Env):
    observation_space: spaces.Space
    action_space: spaces.Space

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            super().reset(seed=seed)
        return self.observation_space.sample(), {}


@pytest.mark.parametrize("model_class", [DQN, PPO, SAC, TD3, TQC])
def test_flatten(model_class) -> None:
    action_space = spaces.Discrete(15) if model_class == DQN else spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
    env = DummyEnv(spaces.Box(-1, 1, shape=(2, 1), dtype=np.float32), action_space)

    model_class("MlpPolicy", env).learn(150)
