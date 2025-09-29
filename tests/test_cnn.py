import numpy as np
import pytest
from stable_baselines3.common.envs import FakeImageEnv

from sbx import DQN, PPO


@pytest.mark.parametrize("model_class", [DQN, PPO])
def test_cnn_dqn(tmp_path, model_class):
    SAVE_NAME = "cnn_model.zip"
    # Fake grayscale with frameskip
    # Atari after preprocessing: 84x84x1, here we are using lower resolution
    # to check that the network handle it automatically
    env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=1)
    kwargs = {}
    if model_class == DQN:
        kwargs = {"buffer_size": 250, "learning_starts": 100}
    elif model_class == PPO:
        kwargs = {
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 2,
        }

    model = model_class(
        "CnnPolicy",
        env,
        policy_kwargs=dict(
            net_arch=[64],
            features_extractor_kwargs=dict(features_dim=64),
        ),
        verbose=1,
        **kwargs
    )
    model.learn(total_timesteps=250)

    obs, _ = env.reset()

    # Test stochastic predict with channel last input
    if model_class == DQN:
        model.exploration_rate = 0.9

    for _ in range(10):
        model.predict(obs, deterministic=False)

    action, _ = model.predict(obs, deterministic=True)

    model.save(tmp_path / SAVE_NAME)
    del model

    model = model_class.load(tmp_path / SAVE_NAME)

    # Check that the prediction is the same
    assert np.allclose(action, model.predict(obs, deterministic=True)[0])

    (tmp_path / SAVE_NAME).unlink()
