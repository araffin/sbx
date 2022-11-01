import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from sbx import PPO, SAC, TQC, DroQ


def test_droq(tmp_path):
    model = DroQ(
        "MlpPolicy",
        "Pendulum-v1",
        learning_starts=50,
        learning_rate=1e-3,
        tau=0.02,
        gamma=0.98,
        verbose=1,
        buffer_size=5000,
        gradient_steps=2,
        ent_coef="auto_1.0",
        seed=1,
        dropout_rate=0.001,
        layer_norm=True,
        # action_noise=NormalActionNoise(np.zeros(1), np.zeros(1)),
    )
    model.learn(total_timesteps=1500)
    # Check that something was learned
    evaluate_policy(model, model.get_env(), reward_threshold=-800)
    model.save(tmp_path / "test_save.zip")
    env = model.get_env()
    obs = env.observation_space.sample()
    action_before = model.predict(obs, deterministic=True)[0]
    # Check we have the same performance
    model = DroQ.load(tmp_path / "test_save.zip")
    evaluate_policy(model, env, reward_threshold=-800)
    action_after = model.predict(obs, deterministic=True)[0]
    assert np.allclose(action_before, action_after)
    # Continue training
    model.set_env(env, force_reset=False)
    model.learn(100, reset_num_timesteps=False)


def test_tqc():
    # Multi env
    train_env = make_vec_env("Pendulum-v1", n_envs=4)
    model = TQC(
        "MlpPolicy",
        train_env,
        top_quantiles_to_drop_per_net=1,
        ent_coef=0.01,
        verbose=1,
        gradient_steps=1,
        use_sde=True,
        qf_learning_rate=1e-3,
    )
    model.learn(200)


@pytest.mark.parametrize("model_class", [SAC])
def test_sac(model_class):
    model = model_class(
        "MlpPolicy",
        "Pendulum-v1",
        verbose=1,
        gradient_steps=1,
        learning_rate=1e-3,
    )
    model.learn(110)


def test_ppo():
    model = PPO(
        "MlpPolicy",
        "Pendulum-v1",
        verbose=1,
        n_steps=64,
        n_epochs=2,
    )
    model.learn(128, progress_bar=False)
