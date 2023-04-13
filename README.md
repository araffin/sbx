<!-- <img src="docs/\_static/img/logo.png" align="right" width="40%"/> -->

<!-- [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) -->
![CI](https://github.com/araffin/sbx/workflows/CI/badge.svg)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Stable Baselines Jax (SB3 + Jax = SBX)

Proof of concept version of [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) in Jax.

Implemented algorithms:
- [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) and [SAC-N](https://arxiv.org/abs/2110.01548)
- [Truncated Quantile Critics (TQC)](https://arxiv.org/abs/2005.04269)
- [Dropout Q-Functions for Doubly Efficient Reinforcement Learning (DroQ)](https://openreview.net/forum?id=xCVJMsPv3RT)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Deep Q Network (DQN)](https://arxiv.org/abs/1312.5602)


### Install using pip

For the latest master version:
```
pip install git+https://github.com/araffin/sbx
```
or:
```
pip install sbx-rl
```

## Example


```python
import gymnasium as gym

from sbx import TQC, DroQ, SAC, PPO, DQN

env = gym.make("Pendulum-v1")

model = TQC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

vec_env.close()
```

## Citing the Project

To cite this repository in publications:

```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

## Maintainers

Stable-Baselines3 is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave), [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli) and [Quentin Gallouédec](https://gallouedec.com/) (@qgallouedec).

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/) or [Stack Overflow](https://stackoverflow.com/) in that case.


## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.
