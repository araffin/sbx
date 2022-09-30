import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from sbx import TQC


def test_tqc(tmp_path):
    model = TQC(
        "MlpPolicy",
        "Pendulum-v1",
        learning_starts=50,
        learning_rate=1e-3,
        tau=0.02,
        gamma=0.98,
        verbose=1,
        buffer_size=5000,
        gradient_steps=2,
        seed=1,
        # action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
    )
    model.learn(total_timesteps=100)
    model.save(tmp_path / "test_save.zip")
    env = model.get_env()
    obs = env.observation_space.sample()
    model = TQC.load(tmp_path / "test_save.zip")
    model.set_env(env, force_reset=False)
    model.learn(1500, reset_num_timesteps=False)
    action_before = model.predict(obs, deterministic=True)[0]
    # Check that something was learned
    evaluate_policy(model, model.get_env(), reward_threshold=-800)
    model.save(tmp_path / "test_save.zip")
    env = model.get_env()
    # Check we have the same performance
    model = TQC.load(tmp_path / "test_save.zip")
    evaluate_policy(model, env, reward_threshold=-800)
    action_after = model.predict(obs, deterministic=True)[0]
    assert np.allclose(action_before, action_after)
