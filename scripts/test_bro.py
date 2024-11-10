from sbx.bro.bro import BRO
import gymnasium as gym

env = gym.make("Pendulum-v1")

model = BRO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)
