from functools import partial
from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union

import gym
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.
    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    observation_space: gym.Space
    _features_dim: int = 0

    def setup(self):
        assert self._features_dim > 0, f"feature dimension must be positive, is {self._features_dim}"

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError()

class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.
    :param observation_space:
    """
    observation_space: gym.Space

    def setup(self):
        super().__init__(self.observation_space, 
                        get_flattened_obs_dim(self.observation_space))


    def forward(self, observations: jnp.ndarray) -> jnp.ndarray:
        return observations.reshape((observations.shape[0], -1))


class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    observation_space: gym.Space
    features_dim: int = 0

   
    # We assume CxHxW images (channels first)
    # Re-ordering will be done by pre-preprocessing or wrapper
    # assert is_image_space(observation_space, check_channels=False), (
    #     "You should use NatureCNN "
    #     f"only with images not with {observation_space}\n"
    #     "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
    #     "If you are using a custom environment,\n"
    #     "please check it using our env checker:\n"
    #     "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
    # )

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4))(x)
        x = nn.relu(x),
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x),
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1))(x)
        x = nn.relu(x),
        x = x.reshape((x.shape[0], -1)) # Flatten
        x = nn.Dense(self.features_dim)(x)
        return self.linear(self.cnn(observations))
