import pickle

import gymnasium as gym
import jax
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sbx import SAC
from sbx.common.gpu_replay_buffer import GpuReplayBuffer, to_training_samples


class DummyEnv(gym.Env):
    """
    Custom gym environment for testing purposes
    """

    def __init__(self):
        self.observation_space = spaces.Box(1, 5, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(1, 5, shape=(1,), dtype=np.float32)
        self._observations = np.arange(1, 6, dtype=np.float32).reshape(-1, 1)
        self._t = 0
        self._ep_length = 100

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._observations[0].copy(), {}

    def step(self, action):
        self._t += 1
        obs = self._observations[self._t % len(self._observations)].copy()
        truncated = self._t >= self._ep_length
        return obs, float(obs[0]), False, truncated, {}


def make_buffer(n_envs: int = 4, **kwargs) -> GpuReplayBuffer:
    """
    Create a replay buffer with continuous observations and actions.
    """
    observation_space = spaces.Box(-1000, 1000, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    return GpuReplayBuffer(32, observation_space, action_space, n_envs=n_envs, **kwargs)


def fill_buffer(replay_buffer: GpuReplayBuffer, n_steps: int) -> None:
    """
    Add transitions with a unique value for each step and environment.
    """
    for step in range(n_steps):
        values = np.arange(replay_buffer.n_envs, dtype=np.float32) + step * replay_buffer.n_envs
        obs = values[:, None] + np.arange(3, dtype=np.float32) / 8
        actions = values[:, None] / 256
        rewards = values / 8
        dones = values % 2
        infos = [{"TimeLimit.truncated": bool(value % 3 == 0)} for value in values]
        replay_buffer.add(obs, obs + 1, actions, rewards, dones, infos)


def test_env():
    check_env(DummyEnv(), warn=False, skip_render_check=True)


@pytest.mark.parametrize("n_envs", [1, 4])
@pytest.mark.parametrize("n_steps", [1, 3, 8, 32, 35])
def test_replay_buffer(n_envs, n_steps):
    replay_buffer = make_buffer(n_envs=n_envs)
    fill_buffer(replay_buffer, n_steps)
    buffer_size = 32 // n_envs

    assert replay_buffer.pos == n_steps % buffer_size
    assert replay_buffer.full == (n_steps >= buffer_size)
    assert replay_buffer.size() == min(n_steps, buffer_size)
    assert isinstance(replay_buffer.observations, jax.Array)
    assert isinstance(replay_buffer.next_observations, jax.Array)

    # Check stored values, including overwritten transitions
    for step in range(max(0, n_steps - buffer_size), n_steps):
        values = np.arange(n_envs, dtype=np.float32) + step * n_envs
        obs = values[:, None] + np.arange(3, dtype=np.float32) / 8
        np.testing.assert_array_equal(replay_buffer.observations[step % buffer_size], obs)
        np.testing.assert_array_equal(replay_buffer.next_observations[step % buffer_size], obs + 1)

    data = replay_buffer.sample(256)
    assert isinstance(data.observations, jax.Array)
    assert isinstance(data.next_observations, jax.Array)
    assert data.observations.shape == (256, 3)
    assert data.next_observations.shape == (256, 3)
    assert data.actions.shape == (256, 1)
    assert data.dones.shape == (256, 1)
    assert data.rewards.shape == (256, 1)

    # All sampled fields must come from the same transition
    values = np.asarray(data.observations)[:, 0]
    assert np.all(values >= max(0, n_steps - buffer_size) * n_envs)
    assert np.all(values < n_steps * n_envs)
    np.testing.assert_array_equal(data.observations, values[:, None] + np.arange(3, dtype=np.float32) / 8)
    np.testing.assert_array_equal(data.next_observations, data.observations + 1)
    np.testing.assert_array_equal(data.actions[:, 0], values / 256)
    np.testing.assert_array_equal(data.rewards[:, 0], values / 8)
    np.testing.assert_array_equal(data.dones[:, 0], (values % 2) * (values % 3 != 0))


def test_replay_buffer_reset():
    replay_buffer = make_buffer()
    with pytest.raises(RuntimeError, match="before any transition"):
        replay_buffer.sample(16)

    fill_buffer(replay_buffer, 10)
    replay_buffer.reset()

    assert replay_buffer.pos == 0
    assert not replay_buffer.full
    assert replay_buffer.size() == 0
    with pytest.raises(RuntimeError, match="before any transition"):
        replay_buffer.sample(16)

    fill_buffer(replay_buffer, 1)
    data = replay_buffer.sample(16)
    assert np.all(np.asarray(data.observations)[:, 0] < replay_buffer.n_envs)


@pytest.mark.parametrize("handle_timeout_termination", [False, True])
def test_replay_buffer_timeouts(handle_timeout_termination):
    replay_buffer = make_buffer(handle_timeout_termination=handle_timeout_termination)
    fill_buffer(replay_buffer, 1)
    data = replay_buffer.sample(64)

    values = np.asarray(data.observations)[:, 0]
    expected_dones = values % 2
    if handle_timeout_termination:
        expected_dones *= values % 3 != 0
    np.testing.assert_array_equal(data.dones[:, 0], expected_dones)


@pytest.mark.parametrize("norm_obs", [False, True])
@pytest.mark.parametrize("norm_reward", [False, True])
def test_replay_buffer_normalization(norm_obs, norm_reward):
    env = VecNormalize(
        make_vec_env(DummyEnv),
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=1.5,
    )
    try:
        replay_buffer = GpuReplayBuffer(100, env.observation_space, env.action_space)

        # Interact and store original observations and rewards
        env.reset()
        obs = env.get_original_obs()
        for _ in range(100):
            action = obs.copy()
            _, _, done, infos = env.step(action)
            next_obs = env.get_original_obs()
            reward = env.get_original_reward()
            replay_buffer.add(obs, next_obs, action, reward, done, infos)
            obs = next_obs

        for _ in range(2):
            data = replay_buffer.sample(64, env=env)

            # Actions identify the original observations in DummyEnv
            obs = data.actions
            next_obs = obs % 5 + 1
            np.testing.assert_allclose(data.observations, env.normalize_obs(obs), atol=1e-6)
            np.testing.assert_allclose(data.next_observations, env.normalize_obs(next_obs), atol=1e-6)
            np.testing.assert_allclose(data.rewards, env.normalize_reward(next_obs), atol=1e-6)

            # Sampling must use the current normalization statistics
            if norm_obs:
                env.obs_rms.mean += 0.5
                env.obs_rms.var *= 2
                env.obs_rms.count += 1
            env.ret_rms.var *= 2
    finally:
        env.close()


@pytest.mark.parametrize("jax_backend", ["cpu", "gpu"])
def test_device_buffer(jax_backend):
    if jax_backend == "gpu":
        try:
            devices = jax.devices("gpu")
        except RuntimeError:
            pytest.skip("JAX GPU not available")
        if not devices:
            pytest.skip("JAX GPU not available")

    replay_buffer = make_buffer(jax_backend=jax_backend)
    fill_buffer(replay_buffer, 3)
    data = replay_buffer.sample(16)
    desired_device = jax.devices(jax_backend)[0]

    # Observations use JAX; the remaining fields use numpy
    for value in [
        replay_buffer.observations,
        replay_buffer.next_observations,
        data.observations,
        data.next_observations,
    ]:
        assert isinstance(value, jax.Array)
        assert value.devices() == {desired_device}

    for value in [data.actions, data.rewards, data.dones]:
        assert isinstance(value, np.ndarray)


@pytest.mark.parametrize("action_dtype", [np.float32, np.float64])
def test_buffer_dtypes(action_dtype):
    observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = spaces.Box(-1, 1, shape=(2, 2), dtype=action_dtype)
    replay_buffer = GpuReplayBuffer(1, observation_space, action_space)

    obs = np.array([[0.0, 0.25, 0.5]], dtype=np.float32)
    action = np.array([[[-1.0, -0.5], [0.5, 1.0]]], dtype=action_dtype)
    replay_buffer.add(obs, obs + 0.25, action, np.ones(1), np.zeros(1), [{}])
    data = replay_buffer.sample(16)

    assert replay_buffer.observations.dtype == np.float32
    assert replay_buffer.next_observations.dtype == np.float32
    assert replay_buffer.actions.dtype == np.float32
    assert data.observations.dtype == np.float32
    assert data.next_observations.dtype == np.float32
    assert data.actions.dtype == np.float32
    assert data.rewards.dtype == np.float32
    assert data.dones.dtype == np.float32
    assert data.actions.shape == (16, 4)
    np.testing.assert_array_equal(data.actions, np.repeat(action.reshape(1, 4), 16, axis=0))


def test_to_training_samples():
    replay_buffer = make_buffer()
    fill_buffer(replay_buffer, 3)
    data = replay_buffer.sample(16)
    training_data = to_training_samples(data, gamma=0.99, n_samples=16)

    # Conversion must not copy observations back to numpy
    assert training_data.observations is data.observations
    assert training_data.next_observations is data.next_observations
    np.testing.assert_array_equal(training_data.actions, data.actions)
    np.testing.assert_array_equal(training_data.dones, data.dones.flatten())
    np.testing.assert_array_equal(training_data.rewards, data.rewards.flatten())
    assert training_data.discounts.shape == (16,)
    np.testing.assert_allclose(training_data.discounts, 0.99)


@pytest.mark.parametrize("n_steps", [3, 35])
def test_replay_buffer_save_load(tmp_path, n_steps):
    replay_buffer = make_buffer()
    fill_buffer(replay_buffer, n_steps)

    with (tmp_path / "replay_buffer.pkl").open("wb") as file:
        pickle.dump(replay_buffer, file)
    with (tmp_path / "replay_buffer.pkl").open("rb") as file:
        loaded_buffer = pickle.load(file)

    assert loaded_buffer.pos == replay_buffer.pos
    assert loaded_buffer.full == replay_buffer.full
    assert loaded_buffer.size() == replay_buffer.size()
    assert isinstance(loaded_buffer.observations, jax.Array)
    assert isinstance(loaded_buffer.next_observations, jax.Array)
    assert loaded_buffer.observations.devices() == replay_buffer.observations.devices()
    assert loaded_buffer.next_observations.devices() == replay_buffer.next_observations.devices()

    for name in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
        np.testing.assert_array_equal(getattr(loaded_buffer, name), getattr(replay_buffer, name))

    # Check that the loaded buffer can still be updated and sampled
    fill_buffer(loaded_buffer, 1)
    data = loaded_buffer.sample(16)
    values = np.asarray(data.observations)[:, 0]
    np.testing.assert_array_equal(data.next_observations, data.observations + 1)
    np.testing.assert_array_equal(data.actions[:, 0], values / 256)
    np.testing.assert_array_equal(data.rewards[:, 0], values / 8)
    np.testing.assert_array_equal(data.dones[:, 0], (values % 2) * (values % 3 != 0))


def test_replay_buffer_optimize_memory_usage():
    with pytest.raises(NotImplementedError, match="optimize_memory_usage"):
        make_buffer(optimize_memory_usage=True)


@pytest.mark.parametrize(
    "observation_space",
    [
        spaces.Box(-1, 1, shape=(3,), dtype=np.float64),
        spaces.Box(-1, 1, shape=(2, 3), dtype=np.float32),
        spaces.Dict(dict(obs=spaces.Box(-1, 1, shape=(3,), dtype=np.float32))),
    ],
)
def test_replay_buffer_observation_space(observation_space):
    action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    with pytest.raises(TypeError, match="GpuReplayBuffer requires"):
        GpuReplayBuffer(32, observation_space, action_space)


def test_sac_n_steps():
    with pytest.raises(ValueError, match="n_steps"):
        SAC("MlpPolicy", "Pendulum-v1", n_steps=2, replay_buffer_class=GpuReplayBuffer)


@pytest.mark.parametrize("replay_buffer_class", [None, GpuReplayBuffer])
def test_sac(replay_buffer_class):
    env = make_vec_env("Pendulum-v1", n_envs=2, seed=0)
    try:
        model = SAC(
            "MlpPolicy",
            env,
            buffer_size=128,
            learning_starts=16,
            batch_size=16,
            gradient_steps=2,
            replay_buffer_class=replay_buffer_class,
            policy_kwargs=dict(net_arch=[16]),
            seed=0,
        )
        model.learn(80)

        assert model._n_updates > 0
        for state in [model.policy.actor_state, model.policy.qf_state]:
            for param in jax.tree_util.tree_leaves(state.params):
                assert np.all(np.isfinite(param))
    finally:
        env.close()


def test_sac_save_load(tmp_path):
    env = make_vec_env("Pendulum-v1", n_envs=2, seed=0)
    try:
        model = SAC(
            "MlpPolicy",
            env,
            buffer_size=128,
            learning_starts=16,
            batch_size=16,
            gradient_steps=2,
            replay_buffer_class=GpuReplayBuffer,
            policy_kwargs=dict(net_arch=[16]),
            seed=0,
        )
        model.learn(80)

        obs = env.reset()
        action_before = model.predict(obs, deterministic=True)[0]
        model.save(tmp_path / "model.zip")
        model.save_replay_buffer(tmp_path / "replay_buffer.pkl")
        replay_buffer = model.replay_buffer

        model = SAC.load(tmp_path / "model.zip", env=env)
        action_after = model.predict(obs, deterministic=True)[0]
        np.testing.assert_allclose(action_before, action_after)

        model.load_replay_buffer(tmp_path / "replay_buffer.pkl")
        assert isinstance(model.replay_buffer, GpuReplayBuffer)
        assert model.replay_buffer.pos == replay_buffer.pos
        assert model.replay_buffer.full == replay_buffer.full
        for name in ["observations", "next_observations", "actions", "rewards", "dones", "timeouts"]:
            np.testing.assert_array_equal(getattr(model.replay_buffer, name), getattr(replay_buffer, name))
        assert isinstance(model.replay_buffer.observations, jax.Array)
        assert isinstance(model.replay_buffer.next_observations, jax.Array)

        # Continue training after loading the replay buffer
        n_updates = model._n_updates
        model.learn(16, reset_num_timesteps=False)
        assert model._n_updates > n_updates
    finally:
        env.close()
