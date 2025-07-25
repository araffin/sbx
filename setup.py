import os

from setuptools import find_packages, setup

with open(os.path.join("sbx", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable Baselines Jax (SB3 + JAX = SBX)

See https://github.com/araffin/sbx

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

## Example

```python
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

model = TQC("MlpPolicy", "Pendulum-v1", verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)

"""


setup(
    name="sbx-rl",
    packages=[package for package in find_packages() if package.startswith("sbx")],
    package_data={"sbx": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3>=2.7.0a1,<3.0",
        "jax>=0.4.24,<0.7.0",  # tf probability not compatible yet with latest jax version
        "jaxlib",
        "flax",
        "optax",
        "tqdm",
        "rich",
        "tensorflow_probability",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "mypy",
            # Lint code
            "ruff>=0.3.1",
            # Reformat
            "black>=25.1.0,<26",
        ],
    },
    description="Jax version of Stable Baselines, implementations of reinforcement learning algorithms.",
    author="Antonin Raffin",
    url="https://github.com/araffin/sbx",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym gymnasium jax openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.9",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
