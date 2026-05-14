import argparse
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from sbx import PPO, RecurrentPPO


def main():
    algo = "ppo"
    algo = "rppo"
    ALGO = RecurrentPPO if algo == "rppo" else PPO
    print(f"Using {algo}")
    
    n_steps = 128
    batch_size = 32 
    train_steps = 20_000
    test_steps = 10
    n_envs = 8

    n_steps = 64
    batch_size = 16 
    train_steps = 20_000
    test_steps = 10
    n_envs = 2

    env_id = "CartPole-v1"

    # create vec env and train algo
    vec_env = make_vec_env(env_id, n_envs=n_envs)
    model = ALGO("MlpPolicy", vec_env, n_steps=n_steps, batch_size=batch_size, verbose=1)
    model.learn(total_timesteps=train_steps, progress_bar=True)

    # test if trained algo works
    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(test_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

    vec_env.close()

if __name__ == "__main__":
    main()