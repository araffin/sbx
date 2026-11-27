"""Replay buffer with observations stored and normalized on the JAX device."""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from sbx.common.type_aliases import ReplayBufferSamplesNp


class GpuReplayBufferSamples(NamedTuple):
    """
    Replay buffer samples with JAX observations and numpy actions, dones and rewards.
    """

    observations: jax.Array
    actions: np.ndarray
    next_observations: jax.Array
    dones: np.ndarray
    rewards: np.ndarray
    discounts: None = None


def to_training_samples(
    data: GpuReplayBufferSamples,
    gamma: float,
    n_samples: int,
) -> ReplayBufferSamplesNp:
    """
    Convert replay buffer samples to the format used by SBX training.
    Observations remain on the JAX device.

    :param data: Replay buffer samples
    :param gamma: Discount factor
    :param n_samples: Number of sampled transitions
    :return: Samples for the training update
    """
    if data.discounts is None:
        discounts = np.full((n_samples,), gamma, dtype=np.float32)
    else:
        discounts = np.asarray(data.discounts).flatten()

    return ReplayBufferSamplesNp(
        data.observations,  # type: ignore[arg-type]
        np.asarray(data.actions),
        data.next_observations,  # type: ignore[arg-type]
        np.asarray(data.dones).flatten(),
        np.asarray(data.rewards).flatten(),
        discounts,
    )


class GpuReplayBuffer(ReplayBuffer):
    """
    Replay buffer used in SBX off-policy algorithms like SAC.

    Observations are stored and normalized on the JAX device.
    Actions, rewards, dones and timeouts are stored as numpy arrays.
    Only one-dimensional float32 Box observations are supported.

    :param buffer_size: Max number of elements in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: Accepted for SB3 compatibility. Storage uses jax_backend.
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
    :param chunk_count: Unused argument retained for compatibility
    :param jax_backend: JAX backend used for observation storage.
        If None, use the default JAX backend.
    """

    observations: jax.Array  # type: ignore[assignment]
    next_observations: jax.Array  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Box | spaces.Discrete,
        device: Any = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        chunk_count: int = 32,
        jax_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(f"GpuReplayBuffer got unexpected arguments: {sorted(kwargs)}")
        if optimize_memory_usage:
            raise NotImplementedError("GpuReplayBuffer does not support optimize_memory_usage=True")
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError(f"GpuReplayBuffer requires a 1-D Box observation space, got {observation_space}")
        if observation_space.dtype != np.float32:
            raise TypeError(f"GpuReplayBuffer requires float32 observations, got {observation_space.dtype}")
        if not isinstance(action_space, spaces.Box | spaces.Discrete):
            raise TypeError(f"GpuReplayBuffer requires a Box or Discrete action space, got {action_space}")
        if int(n_envs) < 1:
            raise ValueError("n_envs must be >= 1")
        if int(chunk_count) < 1:
            raise ValueError("chunk_count must be >= 1")

        # ReplayBuffer.__init__ is deliberately not called: it allocates the whole
        # buffer in host numpy memory, while this class stores the observations
        # on the JAX device. BaseBuffer.__init__ only performs bookkeeping.
        BaseBuffer.__init__(self, buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(int(buffer_size) // self.n_envs, 1)
        self.optimize_memory_usage = optimize_memory_usage

        # Override the PyTorch device set by BaseBuffer: storage uses a JAX device
        self._backend = jax_backend if jax_backend is not None else jax.default_backend()
        self.device = jax.devices(self._backend)[0]
        self.observations = jnp.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=jnp.float32, device=self.device)
        self.next_observations = jnp.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape), dtype=jnp.float32, device=self.device
        )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Reuse device statistics while the running statistics remain unchanged
        self._norm_cache_key: tuple[int, int, float, float, float] | None = None
        self._norm_mean: jax.Array | None = None
        self._norm_scale: jax.Array | None = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # # Copy to avoid modification by reference during the device transfer
        # obs = np.array(obs, dtype=np.float32)
        # next_obs = np.array(next_obs, dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        if obs.shape != (self.n_envs, *self.obs_shape) or next_obs.shape != (self.n_envs, *self.obs_shape):
            raise ValueError(f"Expected obs and next_obs of shape {(self.n_envs, *self.obs_shape)}")

        # Reshape to handle multi-dim and discrete action spaces
        action = np.asarray(action).reshape((self.n_envs, self.action_dim))

        self.observations, self.next_observations = self._add(
            self.observations,
            self.next_observations,
            jax.device_put(np.int32(self.pos), self.device),
            jax.device_put(obs, self.device),
            jax.device_put(next_obs, self.device),
        )
        self.actions[self.pos] = action
        self.rewards[self.pos] = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        self.dones[self.pos] = np.asarray(done, dtype=np.float32).reshape(self.n_envs)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = [info.get("TimeLimit.truncated", False) for info in infos]

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    @staticmethod
    @partial(jax.jit, donate_argnums=(0, 1))
    def _add(
        observations: jax.Array,
        next_observations: jax.Array,
        pos: jax.Array,
        obs: jax.Array,
        next_obs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        # Donate the observation arrays so that JAX can update them in place
        return observations.at[pos].set(obs), next_observations.at[pos].set(next_obs)

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> GpuReplayBufferSamples:  # type: ignore[override]
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of elements to sample
        :param env: Associated VecNormalize environment
            to normalize the observations/rewards when sampling
        :return: Replay buffer samples
        """
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound == 0:
            raise RuntimeError("GpuReplayBuffer.sample() called before any transition was added")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
        self, batch_inds: np.ndarray, env: VecNormalize | None = None
    ) -> GpuReplayBufferSamples:
        # Sample randomly the env idx
        batch_inds = np.asarray(batch_inds, dtype=np.int32)
        env_indices = np.random.randint(0, high=self.n_envs, size=len(batch_inds), dtype=np.int32)

        obs, next_obs = self._get_observations(batch_inds, env_indices, env)
        data = (
            obs,
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return GpuReplayBufferSamples(*data)

    def _get_observations(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray,
        env: VecNormalize | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        mean, scale, clip_obs = self._get_normalization_params(env)

        return self._gather_and_normalize(
            self.observations,
            self.next_observations,
            jax.device_put(batch_inds, self.device),
            jax.device_put(env_indices, self.device),
            mean,
            scale,
            clip_obs,
        )

    @staticmethod
    @jax.jit
    def _gather_and_normalize(
        observations: jax.Array,
        next_observations: jax.Array,
        batch_inds: jax.Array,
        env_indices: jax.Array,
        mean: jax.Array | None,
        scale: jax.Array | None,
        clip_obs: float,
    ) -> tuple[jax.Array, jax.Array]:
        # Gather and normalize both observation batches in the same compiled call
        obs = observations[batch_inds, env_indices, :]
        next_obs = next_observations[batch_inds, env_indices, :]

        if mean is not None:
            # mean and scale are set together by _get_normalization_params
            assert scale is not None
            # Same affine normalization pass as SB3 VecNormalize.normalize_obs
            obs = jnp.clip((obs - mean) * scale, -clip_obs, clip_obs)
            next_obs = jnp.clip((next_obs - mean) * scale, -clip_obs, clip_obs)

        return obs, next_obs

    def _get_normalization_params(
        self,
        env: VecNormalize | None = None,
    ) -> tuple[jax.Array | None, jax.Array | None, float]:
        if env is None or not env.norm_obs:
            return None, None, float("inf")

        obs_rms = env.obs_rms
        # VecNormalize keeps one RunningMeanStd per key for Dict observation spaces;
        # this buffer only supports Box spaces (checked in __init__).
        if isinstance(obs_rms, dict):
            raise TypeError("Dict observation spaces are not supported (Box observations only)")
        clip_obs = float(env.clip_obs)
        epsilon = float(env.epsilon)
        # Cache the mean and scale on the device to avoid repeated transfers.
        # Refresh them when the environment, statistics object, update count, epsilon or clipping limit changes.
        key = (id(env), id(obs_rms), float(obs_rms.count), epsilon, clip_obs)

        if key != self._norm_cache_key:
            mean = np.asarray(obs_rms.mean, dtype=np.float32)
            var = np.asarray(obs_rms.var, dtype=np.float32)
            scale = (1.0 / np.sqrt(var + epsilon)).astype(np.float32)

            # Transfer running statistics once for both observation batches
            self._norm_mean = jax.device_put(mean, self.device)
            self._norm_scale = jax.device_put(scale, self.device)
            self._norm_cache_key = key

        return self._norm_mean, self._norm_scale, clip_obs

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Convert observations to numpy arrays for saving
        state["observations"] = np.asarray(self.observations)
        state["next_observations"] = np.asarray(self.next_observations)
        state["device"] = None
        # Clear normalization caches so they are rebuilt when sampling
        state["_norm_cache_key"] = None
        state["_norm_mean"] = None
        state["_norm_scale"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        # Restore observations on the JAX device
        self.device = jax.devices(self._backend)[0]
        self.observations = jax.device_put(np.asarray(self.observations, dtype=np.float32), self.device)
        self.next_observations = jax.device_put(np.asarray(self.next_observations, dtype=np.float32), self.device)
        # Clear normalization caches so they are rebuilt when sampling
        self._norm_cache_key = None
        self._norm_mean = None
        self._norm_scale = None
