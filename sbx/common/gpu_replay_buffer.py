"""GPU-based replay buffer for SBX off-policy algorithms.

The SB3 ``ReplayBuffer`` stores everything in host RAM (numpy). For
high-dimensional observations, every ``train()`` call then pays for: 
a random row gather on the CPU, one or more
full-width copies (normalization, torch conversion) and a host-to-device
transfer of the whole sample before the jitted update can even start.

``GpuReplayBuffer`` stores the observations and next-observations as flat
jax device arrays and samples entirely on the accelerator:

- ``add()`` publishes one ring slot (all ``n_envs`` rows) on device with a
  donated jitted update (asynchronous H2D, in-place).
- ``sample()`` gathers and normalizes in a single jitted call whose traced
  shapes do not depend on the ring occupancy (compiles once, cache hit
  afterwards).

The small fields (actions, rewards, dones, timeouts) stay on the host;
``sample()`` returns jax arrays for the observations and numpy arrays for
the rest. ``to_training_samples`` converts a sample into the flat format
expected by the jitted ``_train`` loops of the off-policy algorithms,
keeping the observations on device.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from sbx.common.type_aliases import ReplayBufferSamplesNp


class GpuReplayBufferSamples(NamedTuple):
    """Sampled batch.

    Field names mirror SB3 ``ReplayBufferSamples``. ``observations`` and
    ``next_observations`` are jax arrays on the buffer device; the rest are
    numpy arrays on the host. ``discounts`` is ``None`` for 1-step buffers
    (SB3 default).
    """

    observations: jax.Array
    actions: np.ndarray
    next_observations: jax.Array
    dones: np.ndarray
    rewards: np.ndarray
    discounts: None = None


def _normalize_pass(
    obs: jnp.ndarray, mean: jnp.ndarray, scale: jnp.ndarray, skip_mask: jnp.ndarray, clip: float
) -> jnp.ndarray:
    """Affine normalize + clip, with skip regions left raw and *unclipped*.

    Mirrors SB3 ``VecNormalize.normalize_obs`` for the normalized regions and
    ``VecNormalizeSkipSlice`` for the skip regions (bit-exact pass-through:
    there ``mean == 0`` and ``scale == 1``, so ``(x - 0) * 1 == x``).
    """
    normed = (obs - mean) * scale
    return jnp.where(skip_mask, normed, jnp.clip(normed, -clip, clip))


def _publish_row(
    obs_buf: jnp.ndarray, next_obs_buf: jnp.ndarray, row: jnp.ndarray, obs: np.ndarray, next_obs: np.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Write one slot (all ``n_envs`` rows) of the flat ring on device."""
    return obs_buf.at[row].set(obs), next_obs_buf.at[row].set(next_obs)


# Donating both buffers lets XLA update them in place: each publish replaces
# the array objects held by the buffer, and jax keeps the physical buffer
# alive until any in-flight GPU op that still reads it (e.g. the gather of
# the previous train() call) has completed. Stream ordering then guarantees
# the publish can never overwrite data a not-yet-executed gather is about to
# read.
_PUBLISH_ROW = jax.jit(_publish_row, donate_argnums=(0, 1))


@jax.jit
def _gather_and_normalize(
    obs_buf: jnp.ndarray,
    next_obs_buf: jnp.ndarray,
    rows: jnp.ndarray,
    envs: jnp.ndarray,
    mean: jnp.ndarray,
    scale: jnp.ndarray,
    skip_mask: jnp.ndarray,
    clip: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gather one sample from the flat ring and normalize, all on device.

    All traced shapes are fixed by the buffer geometry and the batch size
    (independent of ring occupancy), so this compiles once per
    (num_slots, n_envs, obs_dim, batch size) configuration. Normalization is
    the same affine pass as SB3's ``VecNormalize`` (identity when ``clip``
    is inf and ``mean``/``scale`` are 0/1); ``skip_mask`` regions pass
    through raw and unclipped, like ``VecNormalizeSkipSlice``.
    """
    obs = _normalize_pass(obs_buf[rows, envs], mean, scale, skip_mask, clip)
    next_obs = _normalize_pass(next_obs_buf[rows, envs], mean, scale, skip_mask, clip)
    return obs, next_obs


def to_training_samples(
    data: GpuReplayBufferSamples,
    gamma: float,
    n_samples: int,
) -> ReplayBufferSamplesNp:
    """Convert a sample into the flat format expected by the jitted ``_train`` loops.

    ``observations`` / ``next_observations`` are kept as jax device arrays
    (no host round-trip in front of the jitted update); the small fields are
    flattened numpy arrays, exactly as the stock torch-buffer path produces
    them. ``discounts`` defaults to ``gamma`` (the buffer is 1-step only, so
    ``data.discounts`` is ``None``).
    """
    if data.discounts is None:
        discounts = np.full((n_samples,), gamma, dtype=np.float32)
    else:
        discounts = np.asarray(data.discounts).flatten()
    return ReplayBufferSamplesNp(
        data.observations,
        np.asarray(data.actions),
        data.next_observations,
        np.asarray(data.dones).flatten(),
        np.asarray(data.rewards).flatten(),
        discounts,
    )


def check_n_steps(n_steps: int, name: str) -> None:
    """Guard for the ``GPU*`` algorithm classes: the buffer is 1-step only."""
    if int(n_steps) != 1:
        raise ValueError(f"{name} only supports n_steps=1 (GpuReplayBuffer is a 1-step buffer)")


class GpuReplayBuffer:
    """Replay buffer whose big fields live in JAX device memory.

    Drop-in replacement for ``stable_baselines3.common.buffers.ReplayBuffer``
    for SBX off-policy algorithms (SAC/TQC/TD3/DDPG/DQN): same constructor
    arguments (plus ``norm_skip_slices`` / ``jax_backend``) and the same
    ``add()`` / ``sample()`` calling convention, but ``sample()`` returns
    jax arrays for the observations instead of torch tensors.

    The torch ``device`` argument passed by SBX is ignored on purpose:
    storage always uses the jax default backend (GPU when available), which
    is the whole point of this class. ``chunk_count`` is a legacy argument
    that is accepted and ignored.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Box,
        action_space: spaces.Box | spaces.Discrete,
        device: str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        chunk_count: int = 32,
        norm_skip_slices: Sequence[tuple[int, int]] = (),
        jax_backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        # `device` (a torch device string, e.g. "cpu" from SBX) is accepted
        # for signature compatibility but ignored: storage uses the jax
        # default backend via `jax_backend`.
        del device
        if kwargs:
            raise TypeError(f"GpuReplayBuffer got unexpected arguments: {sorted(kwargs)}")
        if optimize_memory_usage:
            raise NotImplementedError(
                "GpuReplayBuffer v1 stores obs and next_obs explicitly; " "optimize_memory_usage is not supported."
            )
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError(f"GpuReplayBuffer requires a 1-D Box observation space, got {observation_space}")
        if observation_space.dtype != np.float32:
            raise TypeError(f"GpuReplayBuffer requires float32 observations, got {observation_space.dtype}")
        if not isinstance(action_space, spaces.Box | spaces.Discrete):
            raise TypeError(f"GpuReplayBuffer requires a Box or Discrete action space, got {action_space}")
        if int(n_envs) < 1:
            raise ValueError("n_envs must be >= 1")
        if int(chunk_count) < 1:
            # Legacy v1 argument; kept for compatibility, must stay sane.
            raise ValueError("chunk_count must be >= 1")

        self.n_envs = int(n_envs)
        self.obs_dim = int(observation_space.shape[0])
        self.action_dim = int(np.prod(action_space.shape))
        self.handle_timeout_termination = bool(handle_timeout_termination)
        self._norm_skip_slices = tuple((int(s), int(d)) for (s, d) in norm_skip_slices)
        for start, dim in self._norm_skip_slices:
            if start < 0 or dim < 0 or start + dim > self.obs_dim:
                raise ValueError(
                    f"norm_skip_slices region ({start}, +{dim}) lies outside a {self.obs_dim}-dim observation"
                )

        # SB3-compatible ring geometry: `buffer_size` transitions across
        # n_envs parallel envs -> `num_slots` ring positions, each holding
        # n_envs rows.
        self.num_slots = max(int(buffer_size) // self.n_envs, 1)

        self._backend = jax_backend if jax_backend is not None else jax.default_backend()
        self._device = jax.devices(self._backend)[0]

        # Small fields stay on the host (a few MB even for 1M transitions).
        if isinstance(action_space, spaces.Box):
            act_dtype = np.float32 if action_space.dtype == np.float64 else action_space.dtype
        else:
            # Discrete: int64 indices, same as the SB3 ReplayBuffer
            act_dtype = action_space.dtype
        self.actions = np.zeros((self.num_slots, self.n_envs, self.action_dim), dtype=act_dtype)
        self.rewards = np.zeros((self.num_slots, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.num_slots, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.num_slots, self.n_envs), dtype=np.float32)

        # Big fields on device: one flat ring array each (obs + next_obs).
        ring_shape = (self.num_slots, self.n_envs, self.obs_dim)
        self._obs_buf = jnp.zeros(ring_shape, dtype=jnp.float32, device=self._device)
        self._next_obs_buf = jnp.zeros(ring_shape, dtype=jnp.float32, device=self._device)

        # Ring state (same meaning as SB3 BaseBuffer).
        self.pos = 0
        self.full = False

        # Device constants for the normalization pass.
        self._id_mean = jnp.zeros(self.obs_dim, dtype=jnp.float32, device=self._device)
        self._id_scale = jnp.ones(self.obs_dim, dtype=jnp.float32, device=self._device)
        skip = np.zeros(self.obs_dim, dtype=bool)
        for start, dim in self._norm_skip_slices:
            skip[start : start + dim] = True
        self._skip_mask = jnp.asarray(skip, dtype=jnp.bool_, device=self._device)
        self._norm_cache_key: tuple[int, float] | None = None
        self._norm_mean = self._id_mean
        self._norm_scale = self._id_scale
        self._norm_clip = float("inf")

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add one transition per env (same contract as SB3 ``ReplayBuffer.add``)."""
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        if obs.shape != (self.n_envs, self.obs_dim) or next_obs.shape != (self.n_envs, self.obs_dim):
            raise ValueError(
                f"expected obs/next_obs of shape {(self.n_envs, self.obs_dim)}, got {obs.shape}"
            )

        self.actions[self.pos] = np.asarray(action).reshape(self.n_envs, self.action_dim)
        self.rewards[self.pos] = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        self.dones[self.pos] = np.asarray(done, dtype=np.float32).reshape(self.n_envs)
        if self.handle_timeout_termination:
            self.timeouts[self.pos] = [bool(info.get("TimeLimit.truncated", False)) for info in infos]

        # Per-slot device publish (async H2D + donated in-place update,
        # stream-ordered after any in-flight gather of the previous sample).
        # The device row is valid immediately: obs and metadata of a sampled
        # transition always come from the same write.
        self._obs_buf, self._next_obs_buf = _PUBLISH_ROW(
            self._obs_buf, self._next_obs_buf, jnp.int32(self.pos), obs, next_obs
        )

        self.pos += 1
        if self.pos == self.num_slots:
            self.full = True
            self.pos = 0

    def reset(self) -> None:
        """Reset the ring (SB3 ``BaseBuffer.reset`` semantics)."""
        self.pos = 0
        self.full = False

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int, env: Any = None) -> GpuReplayBufferSamples:
        """Sample ``batch_size`` uniform transitions.

        ``env`` follows the SB3 convention: the ``VecNormalize`` wrapper (or
        ``None``). When given, the same normalization SB3 applies at sample
        time is executed on device; regions listed in ``norm_skip_slices``
        are passed through untouched (``VecNormalizeSkipSlice`` behavior).
        """
        upper = self.num_slots if self.full else self.pos
        if upper == 0:
            raise RuntimeError("GpuReplayBuffer.sample() called before any transition was added")
        n = int(batch_size)
        if n < 1:
            raise ValueError("batch_size must be >= 1")

        mean, scale, clip = self._norm_params(env)

        # One flat index vector (uniform slot x uniform env, same
        # distribution as SB3). The traced shapes of the jitted gather below
        # do not depend on `upper`, so it compiles once, ever.
        slots = np.random.randint(0, upper, size=n, dtype=np.int32)
        envs = np.random.randint(0, self.n_envs, size=n, dtype=np.int32)
        obs, next_obs = _gather_and_normalize(
            self._obs_buf,
            self._next_obs_buf,
            jax.device_put(slots, self._device),
            jax.device_put(envs, self._device),
            mean,
            scale,
            self._skip_mask,
            clip,
        )

        # Small host fields, in the same index order as the device gather.
        actions = self.actions[slots, envs, :]
        dones = (self.dones[slots, envs] * (1.0 - self.timeouts[slots, envs])).astype(np.float32)
        rewards = self._normalize_reward(self.rewards[slots, envs].astype(np.float32), env)

        return GpuReplayBufferSamples(
            observations=obs,
            actions=actions,
            next_observations=next_obs,
            dones=dones,
            rewards=rewards,
            discounts=None,
        )

    def _norm_params(self, env: Any) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """Device-side (mean, scale, clip) for the sample-time normalization.

        Refreshed whenever the VecNormalize running stats change (keyed on
        ``obs_rms.count``), mirroring the ``_refresh`` pattern of
        ``VecNormalizeSkipSlice``.
        """
        if env is None or not getattr(env, "norm_obs", True):
            return self._id_mean, self._id_scale, float("inf")
        rms = env.obs_rms
        key = (int(rms.count), float(env.clip_obs))
        if key != self._norm_cache_key:
            mean = np.asarray(rms.mean, dtype=np.float32)
            scale = (1.0 / np.sqrt(np.asarray(rms.var, dtype=np.float32) + float(env.epsilon))).astype(np.float32)
            for start, dim in self._norm_skip_slices:
                mean[start : start + dim] = 0.0
                scale[start : start + dim] = 1.0
            self._norm_mean = jax.device_put(mean, self._device)
            self._norm_scale = jax.device_put(scale, self._device)
            self._norm_clip = float(env.clip_obs)
            self._norm_cache_key = key
        return self._norm_mean, self._norm_scale, self._norm_clip

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Any) -> np.ndarray:
        """Same convention as SB3 ``BaseBuffer._normalize_reward``."""
        if env is not None:
            return env.normalize_reward(reward.reshape(-1, 1)).astype(np.float32).reshape(-1)
        return reward

    # ------------------------------------------------------------------
    # Persistence / debugging
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # JAX device arrays do not pickle portably; move them to the host.
        state["_obs_buf"] = np.asarray(self._obs_buf)
        state["_next_obs_buf"] = np.asarray(self._next_obs_buf)
        state["_device"] = None
        state.pop("_id_mean", None)
        state.pop("_id_scale", None)
        state.pop("_skip_mask", None)
        state.pop("_norm_mean", None)
        state.pop("_norm_scale", None)
        return state

    def __setstate__(self, state: dict) -> None:
        obs_buf = state.pop("_obs_buf")
        next_obs_buf = state.pop("_next_obs_buf")
        self.__dict__.update(state)
        self._device = jax.devices(self._backend)[0]
        self._obs_buf = jax.device_put(np.asarray(obs_buf, dtype=np.float32), self._device)
        self._next_obs_buf = jax.device_put(np.asarray(next_obs_buf, dtype=np.float32), self._device)
        self._id_mean = jnp.zeros(self.obs_dim, dtype=jnp.float32, device=self._device)
        self._id_scale = jnp.ones(self.obs_dim, dtype=jnp.float32, device=self._device)
        skip = np.zeros(self.obs_dim, dtype=bool)
        for start, dim in self._norm_skip_slices:
            skip[start : start + dim] = True
        self._skip_mask = jnp.asarray(skip, dtype=jnp.bool_, device=self._device)
        self._norm_cache_key = None
        self._norm_mean = self._id_mean
        self._norm_scale = self._id_scale
        self._norm_clip = float("inf")

    def _read_all_slots(self) -> tuple[np.ndarray, np.ndarray]:
        """Test/debug helper: full device-to-host dump of the ring.

        Returns ``(obs, next_obs)``, each of shape ``(num_slots, n_envs,
        obs_dim)``.
        """
        return np.asarray(self._obs_buf), np.asarray(self._next_obs_buf)
