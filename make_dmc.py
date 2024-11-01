#adapted from https://github.com/imgeorgiev/dmc2gymnasium

import logging
import numpy as np

from dm_control import suite
from dm_env import specs
from gymnasium.core import Env
from gymnasium.spaces import Box
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation, RescaleAction

def _spec_to_box(spec, dtype=np.float32):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros
        else:
            logging.error("Unrecognized type")
    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs, dtype=np.float32):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0).astype(dtype)


class DMCGym(Env):
    def __init__(
        self,
        env_name,
        task_kwargs={},
        environment_kwargs={},
        #rendering="egl",
        render_height=64,
        render_width=64,
        render_camera_id=0,
        action_repeat=1
    ):
        domain = env_name.split('-')[0]
        task = env_name.split('-')[1]
        self._env = suite.load(
            domain,
            task,
            task_kwargs,
            environment_kwargs,
        )

        # placeholder to allow built in gymnasium rendering
        self.render_mode = "rgb_array"
        self.render_height = render_height
        self.render_width = render_width
        self.render_camera_id = render_camera_id
        
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        self._observation_space = _spec_to_box(self._env.observation_spec().values())
        self._action_space = _spec_to_box([self._env.action_spec()])
        self.action_repeat = action_repeat

        # set seed if provided with task_kwargs
        if "random" in task_kwargs:
            seed = task_kwargs["random"]
            self._observation_space.seed(seed)
            self._action_space.seed(seed)

    def __getattr__(self, name):
        """Add this here so that we can easily access attributes of the underlying env"""
        return getattr(self._env, name)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        """DMC always has a per-step reward range of (0, 1)"""
        return 0, 1
    
    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        action = np.clip(action, -1.0, 1.0)
        reward = 0
        info = {}
        for i in range(self.action_repeat):
            timestep = self._env.step(action)
            observation = _flatten_obs(timestep.observation)
            reward += timestep.reward
            termination = False  # we never reach a goal
            truncation = timestep.last()
            if truncation:
                return observation, reward, termination, truncation, info
        return observation, reward, termination, truncation, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            if not isinstance(seed, np.random.RandomState):
                seed = np.random.RandomState(seed)
            self._env.task._random = seed
        if options:
            logging.warn("Currently doing nothing with options={:}".format(options))
        timestep = self._env.reset()
        observation = _flatten_obs(timestep.observation)
        info = {}
        return observation, info

    def render(self, height=None, width=None, camera_id=None):
        height = height or self.render_height
        width = width or self.render_width
        camera_id = camera_id or self.render_camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
    
    
def make_env_dmc(env_name: str, action_repeat: int = 1) -> Env:
    env = DMCGym(env_name=env_name, action_repeat=action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    env = FlattenObservation(env)
    return env

    