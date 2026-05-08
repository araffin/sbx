import pytest
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ


class CustomReplayBuffer(ReplayBuffer):
    def __init__(self, *args, custom_flag: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_flag = custom_flag


class CustomRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, custom_flag: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_flag = custom_flag


def test_force_cpu_device(tmp_path):
    if not th.cuda.is_available():
        pytest.skip("No CUDA device")
    model = SAC("MlpPolicy", "Pendulum-v1", buffer_size=200)
    assert model.replay_buffer.device == th.device("cpu")
    model.save_replay_buffer(tmp_path / "replay")
    model.load_replay_buffer(tmp_path / "replay")
    assert model.replay_buffer.device == th.device("cpu")


def test_ppo_custom_rollout_buffer():
    model = PPO(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=8,
        batch_size=8,
        rollout_buffer_class=CustomRolloutBuffer,
        rollout_buffer_kwargs={"custom_flag": True},
    )

    assert isinstance(model.rollout_buffer, CustomRolloutBuffer)
    assert model.rollout_buffer.custom_flag


@pytest.mark.parametrize(
    ("model_class", "env_id"),
    [
        (SAC, "Pendulum-v1"),
        (TD3, "Pendulum-v1"),
        (DDPG, "Pendulum-v1"),
        (DQN, "CartPole-v1"),
        (TQC, "Pendulum-v1"),
        (CrossQ, "Pendulum-v1"),
    ],
)
def test_off_policy_custom_replay_buffer(model_class, env_id: str):
    model = model_class(
        "MlpPolicy",
        env_id,
        buffer_size=100,
        replay_buffer_class=CustomReplayBuffer,
        replay_buffer_kwargs={"custom_flag": True},
    )

    assert isinstance(model.replay_buffer, CustomReplayBuffer)
    assert model.replay_buffer.custom_flag
