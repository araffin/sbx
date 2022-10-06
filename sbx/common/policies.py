# import copy
from typing import Dict, Optional, Tuple, Union

import gym
import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose

# from stable_baselines3.common.utils import is_vectorized_observation


class BaseJaxPolicy(BasePolicy):
    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

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
        actions = np.array(actions).reshape((-1,) + self.action_space.shape)

        if isinstance(self.action_space, gym.spaces.Box):
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
            actions = actions.squeeze(axis=0)

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            raise NotImplementedError()
            # # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            # observation = copy.deepcopy(observation)
            # for key, obs in observation.items():
            #     obs_space = self.observation_space.spaces[key]
            #     if is_image_space(obs_space):
            #         obs_ = maybe_transpose(obs, obs_space)
            #     else:
            #         obs_ = np.array(obs)
            #     vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
            #     # Add batch dimension if needed
            #     observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            raise NotImplementedError()
            # # Dict obs need to be handled separately
            # vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # # Add batch dimension if needed
            # observation = observation.reshape((-1,) + self.observation_space.shape)

        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode
