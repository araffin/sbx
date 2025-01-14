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
- [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)
- [Batch Normalization in Deep Reinforcement Learning (CrossQ)](https://openreview.net/forum?id=PczQtTsTIX)
- [Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning (SimBa)](https://openreview.net/forum?id=jXLiDKsuDo)


Note: parameter resets for off-policy algorithms can be activated by passing a list of timesteps to the model constructor (ex: `param_resets=[int(1e5), int(5e5)]` to reset parameters and optimizers after 100_000 and 500_000 timesteps.

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

from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

env = gym.make("Pendulum-v1", render_mode="human")

model = TQC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    vec_env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

vec_env.close()
```

## Using SBX with the RL Zoo

Since SBX shares the SB3 API, it is compatible with the [RL Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), you just need to override the algorithm mapping:

```python
import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
# See note below to use DroQ configuration
# rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    train()
```

Then you can run this script as you would with the RL Zoo:

```
python train.py --algo sac --env HalfCheetah-v4 -params train_freq:4 gradient_steps:4 -P
```

The same goes for the enjoy script:

```python
import rl_zoo3
import rl_zoo3.enjoy
from rl_zoo3.enjoy import enjoy
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

rl_zoo3.ALGOS["ddpg"] = DDPG
rl_zoo3.ALGOS["dqn"] = DQN
# See note below to use DroQ configuration
# rl_zoo3.ALGOS["droq"] = DroQ
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.ALGOS["ppo"] = PPO
rl_zoo3.ALGOS["td3"] = TD3
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["crossq"] = CrossQ
rl_zoo3.enjoy.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    enjoy()
```

## Note about DroQ

[DroQ](https://openreview.net/forum?id=xCVJMsPv3RT) is a special configuration of SAC.

To have the algorithm with the hyperparameters from the paper, you should use (using RL Zoo config format):
```yaml
HalfCheetah-v4:
  n_timesteps: !!float 1e6
  policy: 'MlpPolicy'
  learning_starts: 10000
  gradient_steps: 20
  policy_delay: 20
  policy_kwargs: "dict(dropout_rate=0.01, layer_norm=True)"
```

and then using the RL Zoo script defined above: `python train.py --algo sac --env HalfCheetah-v4 -c droq.yml -P`.

We recommend playing with the `policy_delay` and `gradient_steps` parameters for better speed/efficiency.
Having a higher learning rate for the q-value function is also helpful: `qf_learning_rate: !!float 1e-3`.

Note: when using the DroQ configuration with CrossQ, you should set `layer_norm=False` as there is already batch normalization.

## Note about SimBa

[SimBa](https://openreview.net/forum?id=jXLiDKsuDo) is a special network architecture for off-policy algorithms (SAC, TQC, ...).

Some recommended hyperparameters (tested on MuJoCo and PyBullet environments):
```python
import optax


default_hyperparams = dict(
    n_envs=1,
    n_timesteps=int(1e6),
    policy="SimbaPolicy",
    learning_rate=3e-4,
    # qf_learning_rate=1e-3,
    policy_kwargs={
        "optimizer_class": optax.adamw,
        # "optimizer_kwargs": {"weight_decay": 0.01},
        # Note: here [128] represent a residual block, not just a single layer
        "net_arch": {"pi": [128], "qf": [256, 256]},
        "n_critics": 2,
    },
    learning_starts=10_000,
    # Important: input normalization using VecNormalize
    normalize={"norm_obs": True, "norm_reward": False},
)

hyperparams = {}

# You can also loop gym.registry
for env_id in [
    "HalfCheetah-v4",
    "HalfCheetahBulletEnv-v0",
    "Ant-v4",
]:
    hyperparams[env_id] = default_hyperparams
```

and then using the RL Zoo script defined above: `python train.py --algo tqc --env HalfCheetah-v4 -c simba.py -P`.


## Benchmark

A partial benchmark can be found on [OpenRL Benchmark](https://wandb.ai/openrlbenchmark/sbx) where you can also find several [reports](https://wandb.ai/openrlbenchmark/sbx/reportlist).


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

## Contributors

We would like to thank our contributors: [@jan1854](https://github.com/jan1854).
