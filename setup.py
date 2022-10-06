import os

from setuptools import find_packages, setup

with open(os.path.join("sbx", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable Baselines Jax (SB3 + JAX = SBX)

See https://github.com/araffin/sbx

## Example

```python
from sbx import TQC, DroQ, SAC

model = TQC("MlpPolicy", "Pendulum-v1", verbose=1)
model.learn(total_timesteps=10_000)

"""


setup(
    name="sbx-rl",
    packages=[package for package in find_packages() if package.startswith("sbx")],
    package_data={"sbx": ["py.typed", "version.txt"]},
    install_requires=[
        "stable_baselines3~=1.6.1",
        "jax",
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
            "pytype",
            # Lint code
            "flake8>=3.8",
            # flake8 not compatible with importlib-metadata>5.0
            "importlib-metadata~=4.13",
            # Find likely bugs
            "flake8-bugbear",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
    },
    description="Jax version of Stable Baselines, implementations of reinforcement learning algorithms.",
    author="Antonin Raffin",
    url="https://github.com/araffin/sbx",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.7",
    # PyPI package information.
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
