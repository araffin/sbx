from typing import Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from sbx import DQN


class DummyDictEnv(gym.Env):
    """Custom Environment for testing dictionary observation spaces"""

    def __init__(
        self,
        use_discrete_actions=False,
        channel_last=False,
        nested_dict_obs=False,
        vec_only=False,
    ):
        super().__init__()
        if use_discrete_actions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        N_CHANNELS = 1
        HEIGHT = 36
        WIDTH = 36

        if channel_last:
            obs_shape = (HEIGHT, WIDTH, N_CHANNELS)
        else:
            obs_shape = (N_CHANNELS, HEIGHT, WIDTH)

        self.observation_space = spaces.Dict(
            {
                # Image obs
                "img": spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                # Vector obs
                "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                # Discrete obs
                "discrete": spaces.Discrete(4),
            }
        )

        # For checking consistency with normal MlpPolicy
        if vec_only:
            self.observation_space = spaces.Dict(
                {
                    # Vector obs
                    "vec": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                }
            )

        if nested_dict_obs:
            # Add dictionary observation inside observation space
            self.observation_space.spaces["nested-dict"] = spaces.Dict({"nested-dict-discrete": spaces.Discrete(4)})

    def seed(self, seed=None):
        if seed is not None:
            self.observation_space.seed(seed)

    def step(self, action):
        reward = 0.0
        terminated = truncated = False
        return self.observation_space.sample(), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.observation_space.seed(seed)
        return self.observation_space.sample(), {}

    def render(self):
        pass


@pytest.mark.parametrize("model_class", [DQN])
def test_consistency(model_class):
    """
    Make sure that dict obs with vector only vs using flatten obs is equivalent.
    This ensures notably that the network architectures are the same.
    """
    use_discrete_actions = model_class == DQN
    dict_env = DummyDictEnv(use_discrete_actions=use_discrete_actions, vec_only=True)
    dict_env.seed(10)
    dict_env = gym.wrappers.TimeLimit(dict_env, 100)
    env = gym.wrappers.FlattenObservation(dict_env)
    obs, _ = dict_env.reset()

    n_steps = 256

    kwargs = dict(buffer_size=250, train_freq=8, gradient_steps=1, learning_starts=0)

    dict_model = model_class("MultiInputPolicy", dict_env, gamma=0.5, seed=1, **kwargs)
    action_before_learning_1, _ = dict_model.predict(obs, deterministic=True)
    dict_model.learn(total_timesteps=n_steps)

    normal_model = model_class("MlpPolicy", env, gamma=0.5, seed=1, **kwargs)
    action_before_learning_2, _ = normal_model.predict(obs["vec"], deterministic=True)
    normal_model.learn(total_timesteps=n_steps)

    action_1, _ = dict_model.predict(obs, deterministic=True)
    action_2, _ = normal_model.predict(obs["vec"], deterministic=True)

    assert np.allclose(action_before_learning_1, action_before_learning_2)
    assert np.allclose(action_1, action_2)


@pytest.mark.parametrize("model_class", [DQN])
def test_dict_spaces(model_class):
    env = DummyDictEnv(use_discrete_actions=True)
    env = gym.wrappers.TimeLimit(env, 100)

    n_steps = 256

    model = model_class(
        "MultiInputPolicy", env, buffer_size=250, policy_kwargs=dict(net_arch=[64]), learning_starts=100, verbose=1
    )
    model.learn(total_timesteps=n_steps)

    obs, _ = env.reset()
    model.predict(obs, deterministic=True)


def test_dict_nested():
    """
    Make sure we throw an appropriate error with nested Dict observation spaces
    """
    # Test without manual wrapping to vec-env
    env = DummyDictEnv(nested_dict_obs=True)

    with pytest.raises(NotImplementedError):
        _ = DQN("MultiInputPolicy", env, seed=1)
