# import copy
from typing import Dict, Optional, Tuple, Union, no_type_check

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation


class Flatten(nn.Module):
    """
    Equivalent to PyTorch nn.Flatten() layer.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))


class BaseJaxPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    @jax.jit
    def sample_action(actor_state, obervations, key):
        dist = actor_state.apply_fn(actor_state.params, obervations)
        action = dist.sample(seed=key)
        return action

    @staticmethod
    @jax.jit
    def select_action(actor_state, obervations):
        return actor_state.apply_fn(actor_state.params, obervations).mode()

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            # Minimal dict support: flatten
            keys = list(self.observation_space.keys())
            vectorized_env = is_vectorized_observation(observation[keys[0]], self.observation_space[keys[0]])

            # Add batch dim and concatenate
            observation = np.concatenate(
                [observation[key].reshape(-1, *self.observation_space[key].shape) for key in keys],
                axis=1,
            )
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            # observation = copy.deepcopy(observation)
            # for key, obs in observation.items():
            #     obs_space = self.observation_space.spaces[key]
            #     if is_image_space(obs_space):
            #         obs_ = maybe_transpose(obs, obs_space)
            #     else:
            #         obs_ = np.array(obs)
            #     vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
            #     # Add batch dimension if needed
            #     observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(self.observation_space, spaces.Dict):
            assert isinstance(observation, np.ndarray)
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        assert isinstance(observation, np.ndarray)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode
