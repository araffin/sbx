"""Unit tests for the GPU-based replay buffer and the ``GPU*`` algorithm variants.

Device-agnostic: the tests run against the jax default backend
(GPU when available, CPU otherwise)::

    python -m pytest tests/test_gpu_replay_buffer.py -v
"""

from __future__ import annotations

import pickle

import gymnasium as gym
import jax
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.running_mean_std import RunningMeanStd

from sbx import SAC, GPUDDPG, GPUDQN, GPUSAC, GPUTD3, GPUTQC
from sbx.common.gpu_replay_buffer import GpuReplayBuffer, _gather_and_normalize, to_training_samples


OBS_DIM = 64
N_ENVS = 4
BUFFER_SIZE = 128  # -> 32 slots
CHUNKS = 4  # legacy chunk_count argument (accepted, ignored by the flat ring)


def make_spaces():
    obs = spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
    act = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    return obs, act


def make_buffer(**kw):
    obs, act = make_spaces()
    kw.setdefault("chunk_count", CHUNKS)
    return GpuReplayBuffer(BUFFER_SIZE, obs, act, n_envs=N_ENVS, **kw)


def pattern(slot, env, dim, offset=0.0):
    """Deterministic per-(slot, env, dim) value, decodable from dim 0.

    Built from multiples of ``2**-8`` with magnitude < ``2**15`` (for
    ``offset < 2**7``), i.e. exactly representable in float32, so a stored
    value round-trips bit-exactly and (slot, env) is decodable with no
    tolerance.
    """
    return slot * 4.0 + env * 0.5 + dim * (2.0 ** -8) + offset


def fill_stream(buf, n_adds, offset=0.0):
    """Feed ``n_adds`` deterministic transitions through ``add()``."""
    infos = [{} for _ in range(N_ENVS)]
    num_slots = buf.num_slots
    for step in range(n_adds):
        slot = step % num_slots
        obs = np.stack([pattern(slot, e, np.arange(OBS_DIM)) for e in range(N_ENVS)]).astype(np.float32)
        next_obs = np.stack(
            [pattern(slot, e, np.arange(OBS_DIM), offset=1.0 + offset) for e in range(N_ENVS)]
        ).astype(np.float32)
        action = np.full((N_ENVS, 3), float(step), dtype=np.float32)
        reward = np.full(N_ENVS, step * 0.1, dtype=np.float32)
        done = np.zeros(N_ENVS, dtype=np.float32)
        buf.add(obs, next_obs, action, reward, done, infos)


def decode(v):
    """Recover (slot, env) from the first obs dim of a sample (exact: all
    ops are exact in float32 for the values produced by ``pattern``)."""
    v = np.asarray(v, dtype=np.float32)
    slot = np.floor(v / 4.0).astype(np.int64)
    env = np.floor((v % 4.0) / 0.5).astype(np.int64)
    return slot, env


class TestStorage:
    def test_publish_is_immediate(self):
        # Flat ring: no CPU staging / chunk flush -- the device row must be
        # valid immediately after add() (v1 could lag by up to one chunk).
        buf = make_buffer()
        fill_stream(buf, 1)
        obs, next_obs = buf._read_all_slots()
        expected = np.stack([pattern(0, e, np.arange(OBS_DIM)) for e in range(N_ENVS)]).astype(np.float32)
        np.testing.assert_array_equal(obs[0], expected)
        np.testing.assert_array_equal(next_obs[0, 0], pattern(0, 0, np.arange(OBS_DIM), offset=1.0).astype(np.float32))
        # ...and immediately readable through sample().
        s = buf.sample(64)
        slot, env = decode(s.observations[:, 0])
        np.testing.assert_array_equal(slot, 0)
        np.testing.assert_array_equal(s.observations[:, 0], pattern(0, env, 0).astype(np.float32))

    def test_roundtrip_full_lap(self):
        buf = make_buffer()
        fill_stream(buf, buf.num_slots)
        assert buf.full and buf.pos == 0
        obs, next_obs = buf._read_all_slots()
        for slot in range(buf.num_slots):
            for e in range(N_ENVS):
                np.testing.assert_array_equal(obs[slot, e], pattern(slot, e, np.arange(OBS_DIM)).astype(np.float32))
                np.testing.assert_array_equal(
                    next_obs[slot, e], pattern(slot, e, np.arange(OBS_DIM), offset=1.0).astype(np.float32)
                )

    def test_ring_overwrite(self):
        buf = make_buffer()
        fill_stream(buf, buf.num_slots)
        # fill_stream applies `offset` to next_obs only -> the overwrite is
        # visible in next_obs (obs values repeat the first lap by design).
        fill_stream(buf, buf.num_slots, offset=100000.0)
        assert buf.full
        _, next_obs = buf._read_all_slots()
        for slot in range(buf.num_slots):
            for e in range(N_ENVS):
                # dim 0, offset lap: value is an exact f32 number
                # (fill_stream adds the base +1.0 to next_obs).
                np.testing.assert_array_equal(next_obs[slot, e, 0], pattern(slot, e, 0, offset=100001.0))

    def test_reset_blocks_empty_sample(self):
        buf = make_buffer()
        fill_stream(buf, buf.num_slots)
        buf.reset()
        assert not buf.full and buf.pos == 0
        with pytest.raises(RuntimeError):
            buf.sample(4)


class TestDiscreteActions:
    def test_discrete_roundtrip(self):
        # Discrete action space (DQN): stored as int64 (num_slots, n_envs, 1)
        # and sampled back in the same (slot, env) order as the obs.
        obs = spaces.Box(-10.0, 10.0, (OBS_DIM,), dtype=np.float32)
        act = spaces.Discrete(5)
        buf = GpuReplayBuffer(BUFFER_SIZE, obs, act, n_envs=N_ENVS)
        infos = [{} for _ in range(N_ENVS)]
        for step in range(buf.num_slots):
            slot = step % buf.num_slots
            obs_ = np.stack([pattern(slot, e, np.arange(OBS_DIM)) for e in range(N_ENVS)]).astype(np.float32)
            next_obs_ = np.stack(
                [pattern(slot, e, np.arange(OBS_DIM), offset=1.0) for e in range(N_ENVS)]
            ).astype(np.float32)
            action = (np.arange(N_ENVS) + step) % 5
            reward = np.full(N_ENVS, step * 0.1, dtype=np.float32)
            done = np.zeros(N_ENVS, dtype=np.float32)
            buf.add(obs_, next_obs_, action, reward, done, infos)
        s = buf.sample(4096)
        assert s.actions.shape == (4096, 1)
        assert s.actions.dtype == np.int64
        slot, env = decode(s.observations[:, 0])
        np.testing.assert_array_equal(s.actions[:, 0], (env + slot) % 5)
        np.testing.assert_array_equal(
            s.next_observations[:, 0], pattern(slot, env, 0, offset=1.0).astype(np.float32)
        )


class TestSampling:
    def test_sample_values_match_stored_slots(self):
        buf = make_buffer()
        fill_stream(buf, buf.num_slots)
        s = buf.sample(4096)
        assert s.observations.shape == (4096, OBS_DIM)
        assert s.next_observations.shape == (4096, OBS_DIM)
        assert s.actions.shape == (4096, 3)
        assert s.dones.shape == (4096,)
        assert s.rewards.shape == (4096,)

        slot, env = decode(s.observations[:, 0])
        # next_obs / rewards / actions must come from the same (slot, env)
        np.testing.assert_array_equal(
            s.next_observations[:, 0], pattern(slot, env, 0, offset=1.0).astype(np.float32)
        )
        # rewards are stored as f32(step * 0.1); 0.1 is not exact in binary.
        np.testing.assert_allclose(s.rewards, slot * 0.1, atol=1e-5)
        np.testing.assert_allclose(s.actions[:, 0], slot, atol=1e-5)
        # uniform draw should hit (at least) most slots
        assert len(np.unique(slot)) > buf.num_slots // 2

    def test_sample_warmup(self):
        buf = make_buffer()
        half = buf.num_slots // 2  # 16 slots written; sampling must stay inside them
        fill_stream(buf, half)
        assert not buf.full and buf.pos == half
        s = buf.sample(1024)
        slot, env = decode(s.observations[:, 0])
        assert int(slot.max()) < half  # never samples un-written slots
        np.testing.assert_array_equal(s.next_observations[:, 0], pattern(slot, env, 0, offset=1.0).astype(np.float32))

    def test_sample_timeout_masking(self):
        buf = make_buffer()
        infos = [{"TimeLimit.truncated": False} for _ in range(N_ENVS)]
        for step in range(buf.num_slots):
            slot = step % buf.num_slots
            obs = np.stack([pattern(slot, e, np.arange(OBS_DIM)) for e in range(N_ENVS)]).astype(np.float32)
            next_obs = np.stack(
                [pattern(slot, e, np.arange(OBS_DIM), offset=1.0) for e in range(N_ENVS)]
            ).astype(np.float32)
            action = np.zeros((N_ENVS, 3), np.float32)
            reward = np.ones(N_ENVS, np.float32)
            done = np.full(N_ENVS, float(slot % 2 == 0), np.float32)
            truncated = bool(slot % 4 == 0)
            infos = [{"TimeLimit.truncated": truncated} for _ in range(N_ENVS)]
            buf.add(obs, next_obs, action, reward, done, infos)
        s = buf.sample(4096)
        slot, _ = decode(s.observations[:, 0])
        expected = np.where(slot % 2 == 0, 1.0, 0.0) * (1.0 - np.where(slot % 4 == 0, 1.0, 0.0))
        np.testing.assert_allclose(s.dones, expected, atol=1e-6)


class TestNormalization:
    def _fake_env(self, seed=0, count=1000, clip_obs=10.0, norm_obs=True):
        rng = np.random.default_rng(seed)
        mean = rng.normal(size=OBS_DIM).astype(np.float32)
        var = (np.abs(rng.normal(size=OBS_DIM)) + 0.1).astype(np.float32)

        class _FakeVecNorm:
            def __init__(self):
                self.obs_rms = RunningMeanStd()
                self.obs_rms.mean = mean.copy()
                self.obs_rms.var = var.copy()
                self.obs_rms.count = count
                self.clip_obs = clip_obs
                self.epsilon = 1e-8
                self.norm_obs = norm_obs
                self.ret_rms = RunningMeanStd()
                self.ret_rms.mean = np.zeros(1, np.float32)
                self.ret_rms.var = np.ones(1, np.float32)
                self.ret_rms.count = count

            def normalize_reward(self, reward):
                return (reward - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + self.epsilon)

        return _FakeVecNorm()

    def test_jit_kernel_matches_cpu_normalization(self):
        rng = np.random.default_rng(0)
        obs_dim = 16
        n_slots, n_envs = 8, 4
        mean = rng.normal(size=obs_dim).astype(np.float32)
        var = (np.abs(rng.normal(size=obs_dim)) + 0.1).astype(np.float32)
        scale = (1.0 / np.sqrt(var + 1e-8)).astype(np.float32)
        buf_np = rng.normal(size=(n_slots, n_envs, obs_dim)).astype(np.float32)
        rows = np.array([0, 7, 3, 5, 2], np.int32)
        envs = np.array([0, 3, 1, 2, 0], np.int32)

        out_obs, out_next = _gather_and_normalize(
            jax.device_put(buf_np),
            jax.device_put(buf_np),
            jax.device_put(rows),
            jax.device_put(envs),
            jax.device_put(mean),
            jax.device_put(scale),
            jax.device_put(np.zeros(obs_dim, dtype=bool)),
            10.0,
        )
        expect = np.clip((buf_np[rows, envs] - mean) * scale, -10.0, 10.0).astype(np.float32)
        np.testing.assert_allclose(out_obs, expect, atol=1e-5)
        np.testing.assert_allclose(out_next, expect, atol=1e-5)

    def test_jit_kernel_skip_slice_not_clipped(self):
        # VecNormalizeSkipSlice semantics: the skip region passes through
        # raw and unclipped, the rest is normalized + clipped.
        rng = np.random.default_rng(2)
        obs_dim = 16
        n_slots, n_envs = 4, 2
        skip = np.zeros(obs_dim, dtype=bool)
        skip[4:8] = True
        mean = rng.normal(size=obs_dim).astype(np.float32)
        var = (np.abs(rng.normal(size=obs_dim)) + 0.1).astype(np.float32)
        scale = (1.0 / np.sqrt(var + 1e-8)).astype(np.float32)
        mean[4:8], scale[4:8] = 0.0, 1.0  # as _norm_params would set them
        # Values in the skip region far outside clip; the rest large too.
        buf_np = np.where(
            skip[None, None, :],
            25.0 + rng.normal(size=(n_slots, n_envs, obs_dim)),
            4.0 + rng.normal(size=(n_slots, n_envs, obs_dim)),
        ).astype(np.float32)
        rows = np.array([0, 1, 3], np.int32)
        envs = np.array([0, 1, 0], np.int32)
        out_obs, _ = _gather_and_normalize(
            jax.device_put(buf_np),
            jax.device_put(buf_np),
            jax.device_put(rows),
            jax.device_put(envs),
            jax.device_put(mean),
            jax.device_put(scale),
            jax.device_put(skip),
            10.0,
        )
        raw = buf_np[rows, envs].copy()
        expect = raw.copy()
        expect[:, ~skip] = np.clip((expect[:, ~skip] - mean[~skip]) * scale[~skip], -10.0, 10.0)
        np.testing.assert_allclose(out_obs, expect, atol=1e-5)
        # Skip region must be exactly the raw stored values (>10, i.e. not clipped).
        np.testing.assert_array_equal(np.asarray(out_obs)[:, 4:8], raw[:, 4:8])

    def test_jit_kernel_identity_pass(self):
        rng = np.random.default_rng(1)
        buf_np = rng.normal(size=(4, 2, 8)).astype(np.float32)
        rows = np.array([1, 3, 0], np.int32)
        envs = np.array([0, 1, 1], np.int32)
        out_obs, _ = _gather_and_normalize(
            jax.device_put(buf_np),
            jax.device_put(buf_np),
            jax.device_put(rows),
            jax.device_put(envs),
            jax.device_put(np.zeros(8, np.float32)),
            jax.device_put(np.ones(8, np.float32)),
            jax.device_put(np.zeros(8, dtype=bool)),
            float("inf"),
        )
        np.testing.assert_array_equal(out_obs, buf_np[rows, envs])

    def test_norm_params_and_refresh(self):
        env = self._fake_env()
        buf = make_buffer()
        mean, scale, clip = buf._norm_params(env)
        np.testing.assert_allclose(np.asarray(mean), env.obs_rms.mean, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(scale), 1.0 / np.sqrt(env.obs_rms.var + env.epsilon), atol=1e-5
        )
        assert clip == env.clip_obs
        # Stats change -> refresh must pick up the new values.
        env.obs_rms.mean = env.obs_rms.mean + 1.0
        env.obs_rms.count += 1
        mean2, _, _ = buf._norm_params(env)
        np.testing.assert_allclose(np.asarray(mean2), env.obs_rms.mean, atol=1e-6)

    def test_norm_skip_slices(self):
        env = self._fake_env()
        buf = make_buffer(norm_skip_slices=((16, 32),))
        mean, scale, _ = buf._norm_params(env)
        mean, scale = np.asarray(mean), np.asarray(scale)
        np.testing.assert_array_equal(mean[16:48], 0.0)
        np.testing.assert_array_equal(scale[16:48], 1.0)
        np.testing.assert_allclose(mean[:16], env.obs_rms.mean[:16], atol=1e-6)
        np.testing.assert_allclose(mean[48:], env.obs_rms.mean[48:], atol=1e-6)

    def test_reward_normalization(self):
        # norm_obs=False keeps the raw (decodable) obs; rewards are still
        # normalized through the env's normalize_reward.
        env = self._fake_env(norm_obs=False)
        buf = make_buffer()
        fill_stream(buf, buf.num_slots)
        s = buf.sample(256, env=env)
        # ret_rms of the fake env is the identity, so normalized reward is
        # (raw - 0) / sqrt(1 + 1e-8); decode the raw reward from the sample.
        slot, _ = decode(s.observations[:, 0])
        expected = (slot * 0.1) / np.sqrt(1.0 + 1e-8)
        np.testing.assert_allclose(s.rewards, expected, atol=1e-5)


class TestToTrainingSamples:
    def test_keeps_jax_obs_and_flattens_rest(self):
        buf = make_buffer()
        fill_stream(buf, buf.num_slots)
        s = buf.sample(64)
        data = to_training_samples(s, gamma=0.99, n_samples=64)
        # obs / next_obs stay jax device arrays (the zero-copy part)
        assert isinstance(data.observations, jax.Array)
        assert isinstance(data.next_observations, jax.Array)
        np.testing.assert_array_equal(np.asarray(data.observations), np.asarray(s.observations))
        np.testing.assert_array_equal(np.asarray(data.next_observations), np.asarray(s.next_observations))
        # small fields are flattened numpy
        assert data.actions.dtype == np.float32
        assert data.actions.shape == (64, 3)
        assert data.dones.shape == (64,)
        assert data.rewards.shape == (64,)
        assert data.discounts.shape == (64,)
        # 1-step buffer: discounts default to gamma
        np.testing.assert_allclose(data.discounts, 0.99)


class TestPersistence:
    def test_pickle_roundtrip(self):
        # 140 transitions / 4 envs -> 35 slots; two laps exercise the
        # ring overwrite. The device arrays are dumped to host on pickle
        # and re-uploaded on unpickle.
        obs, act = make_spaces()
        buf = GpuReplayBuffer(140, obs, act, n_envs=N_ENVS, chunk_count=4)
        fill_stream(buf, buf.num_slots * 2)
        assert buf.full and buf.pos == 0
        obs0, next0 = buf._read_all_slots()

        buf2 = pickle.loads(pickle.dumps(buf))
        assert buf2.full == buf.full and buf2.pos == buf.pos
        obs1, next1 = buf2._read_all_slots()
        np.testing.assert_array_equal(obs0, obs1)
        np.testing.assert_array_equal(next0, next1)
        # Usable after unpickle.
        s = buf2.sample(64)
        assert s.observations.shape == (64, OBS_DIM)


class TestConstructorGuards:
    def test_optimize_memory_usage_rejected(self):
        obs, act = make_spaces()
        with pytest.raises(NotImplementedError):
            GpuReplayBuffer(128, obs, act, n_envs=N_ENVS, optimize_memory_usage=True)

    def test_non_f32_obs_rejected(self):
        obs = spaces.Box(-1, 1, (OBS_DIM,), dtype=np.float64)
        act = spaces.Box(-1, 1, (3,), dtype=np.float32)
        with pytest.raises(TypeError):
            GpuReplayBuffer(128, obs, act, n_envs=N_ENVS)

    def test_dict_obs_rejected(self):
        obs = spaces.Dict({"a": spaces.Box(-1, 1, (4,), np.float32)})
        act = spaces.Box(-1, 1, (3,), dtype=np.float32)
        with pytest.raises(TypeError):
            GpuReplayBuffer(128, obs, act, n_envs=N_ENVS)

    def test_multi_discrete_actions_rejected(self):
        obs, _ = make_spaces()
        act = spaces.MultiDiscrete([2, 3])
        with pytest.raises(TypeError):
            GpuReplayBuffer(128, obs, act, n_envs=N_ENVS)

    def test_unknown_kwargs_rejected(self):
        obs, act = make_spaces()
        with pytest.raises(TypeError):
            GpuReplayBuffer(128, obs, act, n_envs=N_ENVS, totally_unknown=True)

    def test_skip_slice_out_of_bounds_rejected(self):
        obs, act = make_spaces()
        with pytest.raises(ValueError):
            GpuReplayBuffer(128, obs, act, n_envs=N_ENVS, norm_skip_slices=[(48, 32)])


def _make_dummy_env(obs_dim: int):
    class DummyEnv(gym.Env):
        observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)

        def reset(self, *, seed=None, options=None):
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            return self.np_random.uniform(-1, 1, size=obs_dim).astype(np.float32), {}

        def step(self, action):
            obs = self.np_random.uniform(-1, 1, size=obs_dim).astype(np.float32)
            return obs, float(self.np_random.random()), False, False, {}

    return DummyEnv()


def _make_dummy_discrete_env(obs_dim: int, n_actions: int = 4):
    class DummyEnv(gym.Env):
        observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        action_space = spaces.Discrete(n_actions)

        def reset(self, *, seed=None, options=None):
            self.np_random, _ = gym.utils.seeding.np_random(seed)
            return self.np_random.uniform(-1, 1, size=obs_dim).astype(np.float32), {}

        def step(self, action):
            obs = self.np_random.uniform(-1, 1, size=obs_dim).astype(np.float32)
            return obs, float(self.np_random.random()), False, False, {}

    return DummyEnv()


def _assert_finite_params(agent):
    from jax import tree_util

    policy = agent.policy
    # SAC/TD3/DDPG/DQN expose qf_state, TQC exposes qf1_state + qf2_state
    state_names = [n for n in ("qf_state", "qf1_state", "qf2_state") if hasattr(policy, n)]
    assert state_names, f"no qf state found on {type(policy).__name__}"
    for name in state_names:
        leaves = tree_util.tree_leaves(getattr(policy, name).params)
        assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


class TestGPUClassGuards:
    def test_n_steps_rejected(self):
        env = _make_dummy_env(8)
        with pytest.raises(ValueError):
            GPUSAC("MlpPolicy", env, n_steps=2)
        with pytest.raises(ValueError):
            GPUDQN("MlpPolicy", _make_dummy_discrete_env(8), n_steps=2)


class TestGPUSACEndToEnd:
    def test_learn_smoke(self):
        agent = GPUSAC(
            "MlpPolicy",
            _make_dummy_env(8),
            learning_rate=1e-3,
            buffer_size=512,
            batch_size=16,
            gradient_steps=8,
            learning_starts=32,
            train_freq=(1, "step"),
            replay_buffer_kwargs={"chunk_count": 4},
        )
        agent.learn(256)

        assert agent._n_updates > 0
        _assert_finite_params(agent)


class TestGPUTD3EndToEnd:
    def test_learn_smoke(self):
        agent = GPUTD3(
            "MlpPolicy",
            _make_dummy_env(8),
            learning_rate=1e-3,
            buffer_size=512,
            batch_size=16,
            gradient_steps=8,
            learning_starts=32,
            train_freq=(1, "step"),
        )
        agent.learn(256)

        assert agent._n_updates > 0
        _assert_finite_params(agent)


class TestGPUTQCEndToEnd:
    def test_learn_smoke(self):
        agent = GPUTQC(
            "MlpPolicy",
            _make_dummy_env(8),
            learning_rate=1e-3,
            buffer_size=512,
            batch_size=16,
            gradient_steps=8,
            learning_starts=32,
            train_freq=(1, "step"),
        )
        agent.learn(256)

        assert agent._n_updates > 0
        _assert_finite_params(agent)


class TestGPUDDPGEndToEnd:
    def test_learn_smoke(self):
        agent = GPUDDPG(
            "MlpPolicy",
            _make_dummy_env(8),
            learning_rate=1e-3,
            buffer_size=512,
            batch_size=16,
            gradient_steps=8,
            learning_starts=32,
            train_freq=(1, "step"),
        )
        agent.learn(256)

        assert agent._n_updates > 0
        _assert_finite_params(agent)


class TestGPUDQNEndToEnd:
    def test_learn_smoke(self):
        agent = GPUDQN(
            "MlpPolicy",
            _make_dummy_discrete_env(8),
            learning_rate=1e-3,
            buffer_size=512,
            batch_size=16,
            gradient_steps=4,
            learning_starts=32,
            train_freq=(1, "step"),
        )
        agent.learn(128)

        assert agent._n_updates > 0
        _assert_finite_params(agent)


class TestStockBufferRegression:
    # The GPU* train() dispatch must not alter the stock torch-buffer path.
    def test_learn_smoke_stock_sac(self):
        agent = SAC(
            "MlpPolicy",
            _make_dummy_env(8),
            learning_rate=1e-3,
            buffer_size=512,
            batch_size=16,
            gradient_steps=8,
            learning_starts=32,
            train_freq=(1, "step"),
        )
        agent.learn(256)

        assert agent._n_updates > 0
        _assert_finite_params(agent)
