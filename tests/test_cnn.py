import numpy as np
import pytest
from stable_baselines3.common.envs import FakeImageEnv

from sbx import DQN


@pytest.mark.parametrize("model_class", [DQN])
def test_cnn(tmp_path, model_class):
    SAVE_NAME = "cnn_model.zip"
    # Fake grayscale with frameskip
    # Atari after preprocessing: 84x84x1, here we are using lower resolution
    # to check that the network handle it automatically
    env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=1)
    model = model_class(
        "CnnPolicy",
        env,
        buffer_size=250,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=100,
        verbose=1,
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
