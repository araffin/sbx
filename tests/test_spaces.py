from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pytest

from sbx import PPO

BOX_SPACE_FLOAT32 = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


@dataclass
class DummyEnv(gym.Env):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space

    def step(self, action):
        assert action in self.action_space
        return self.observation_space.sample(), 0.0, False, False, {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
        return self.observation_space.sample(), {}


class DummyMultiDiscreteAction(DummyEnv):
    def __init__(self):
        super().__init__(
            BOX_SPACE_FLOAT32,
            gym.spaces.MultiDiscrete([2, 3]),
        )


class DummyMultiBinaryAction(DummyEnv):
    def __init__(self):
        super().__init__(
            BOX_SPACE_FLOAT32,
            gym.spaces.MultiBinary(2),
        )


@pytest.mark.parametrize("env", ["Pendulum-v1", "CartPole-v1", DummyMultiDiscreteAction(), DummyMultiBinaryAction()])
def test_ppo_action_spaces(env):
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=16)
    model.learn(64)


def test_ppo_multidim_discrete_not_supported():
    env = DummyEnv(BOX_SPACE_FLOAT32, gym.spaces.MultiDiscrete([[2, 3]]))
    with pytest.raises(
        AssertionError,
        match=r"Only one-dimensional MultiDiscrete action spaces are supported, but found MultiDiscrete\(.*\).",
    ):
        PPO("MlpPolicy", env)


def test_ppo_multidim_binary_not_supported():
    env = DummyEnv(BOX_SPACE_FLOAT32, gym.spaces.MultiBinary([2, 3]))
    with pytest.raises(AssertionError, match=r"Multi-dimensional MultiBinary\(.*\) action space is not supported"):
        PPO("MlpPolicy", env)
