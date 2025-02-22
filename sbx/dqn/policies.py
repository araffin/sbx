from collections import OrderedDict
from typing import Any, Callable, Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import is_vectorized_observation

from sbx.common.policies import BaseJaxPolicy, Flatten, OneHot
from sbx.common.type_aliases import RLTrainState


class QNetwork(nn.Module):
    n_actions: int
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_actions)(x)
        return x


# Add CNN policy from DQN paper
class NatureCNN(nn.Module):
    n_actions: int
    n_units: int = 512
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Convert from channel-first (PyTorch) to channel-last (Jax)
        x = jnp.transpose(x, (0, 2, 3, 1))
        # Convert to float and normalize the image
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = self.activation_fn(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = self.activation_fn(x)
        # Flatten
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_actions)(x)
        return x


class DQNPolicy(BaseJaxPolicy):
    action_space: spaces.Discrete  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is not None:
            assert isinstance(net_arch, list)
            self.n_units = net_arch[0]
        else:
            self.n_units = 256
        self.activation_fn = activation_fn

    def build(self, key: jax.Array, lr_schedule: Schedule) -> jax.Array:
        key, qf_key = jax.random.split(key, 2)

        obs = jnp.array([self.observation_space.sample()])

        self.qf: nn.Module = QNetwork(
            n_actions=int(self.action_space.n),
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs),
            target_params=self.qf.init({"params": qf_key}, obs),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf.apply = jax.jit(self.qf.apply)  # type: ignore[method-assign]

        return key

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    @staticmethod
    @jax.jit
    def select_action(qf_state, observations):
        qf_values = qf_state.apply_fn(qf_state.params, observations)
        return jnp.argmax(qf_values, axis=1).reshape(-1)

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:  # type: ignore[override]
        return DQNPolicy.select_action(self.qf_state, observation)


class CNNPolicy(DQNPolicy):
    def build(self, key: jax.Array, lr_schedule: Schedule) -> jax.Array:
        key, qf_key = jax.random.split(key, 2)

        obs = jnp.array([self.observation_space.sample()])

        self.qf = NatureCNN(
            n_actions=int(self.action_space.n),
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs),
            target_params=self.qf.init({"params": qf_key}, obs),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )
        self.qf.apply = jax.jit(self.qf.apply)  # type: ignore[method-assign]

        return key


class MultiInputQNetwork(nn.Module):
    observation_space: spaces.Space
    n_actions: int
    cnn_output_dim: int = 256
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def setup(self):
        def layer(subspace):
            if is_image_space(subspace):
                return NatureCNN(
                    n_actions=self.cnn_output_dim,
                    n_units=self.n_units,
                    activation_fn=self.activation_fn,
                )
            elif isinstance(subspace, spaces.Discrete):
                return OneHot(num_classes=subspace.n)
            return Flatten()

        self.extractors = jax.tree_map(layer, self.observation_space.spaces)

    @nn.compact
    def __call__(self, observations: dict[str, jnp.ndarray]) -> jnp.ndarray:
        encoded_tensors = jax.tree_map(lambda extractor, x: extractor(x), self.extractors, flax.core.freeze(observations))

        flattened, _ = jax.tree.flatten(encoded_tensors)
        x = jax.lax.concatenate(flattened, dimension=1)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_actions)(x)
        return x


class MultiInputPolicy(DQNPolicy):
    def build(self, key: jax.Array, lr_schedule: Schedule) -> jax.Array:
        key, qf_key = jax.random.split(key, 2)

        # add batch dimension to the observation values.
        obs = jax.tree.map(lambda x: np.array([x]), self.observation_space.sample())

        self.qf = MultiInputQNetwork(
            observation_space=self.observation_space,
            n_actions=int(self.action_space.n),
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs),
            target_params=self.qf.init({"params": qf_key}, obs),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf.apply = jax.jit(self.qf.apply)  # type: ignore[method-assign]

        return key

    def prepare_obs(  # type: ignore[override]
        self, observation: Union[np.ndarray, dict[str, np.ndarray]]
    ) -> tuple[Union[np.ndarray, dict[str, np.ndarray]], bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(
                self.observation_space, spaces.Dict
            ), f"The observation provided is a dict but the obs space is {self.observation_space}"

            vectorized_env = is_vectorized_observation(observation, self.observation_space)  # type: ignore[arg-type]
            observation = jax.tree.map(
                lambda obs, obs_space: self._prepare_obs(obs, obs_space)[0],
                OrderedDict(observation),
                OrderedDict(self.observation_space.spaces),
            )
            return observation, vectorized_env

        return super().prepare_obs(observation)
