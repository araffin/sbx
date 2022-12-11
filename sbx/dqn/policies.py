from typing import Any, Callable, Dict, List, Optional, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import BaseJaxPolicy
from sbx.common.type_aliases import RLTrainState


class QNetwork(nn.Module):
    n_actions: int
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        return x


class DQNPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
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

    def build(self, key, lr_schedule: Schedule) -> None:
        key, qf_key = jax.random.split(key, 2)

        obs = jnp.array([self.observation_space.sample()])

        self.qf = QNetwork(
            n_actions=self.action_space.n,
            n_units=self.n_units,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs),
            target_params=self.qf.init({"params": qf_key}, obs),
            tx=self.optimizer_class(learning_rate=lr_schedule(1), **self.optimizer_kwargs),
        )

        # TODO: jit qf.apply_fn too?
        self.qf.apply = jax.jit(self.qf.apply)

        return key

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    @staticmethod
    @jax.jit
    def select_action(qf_state, observations):
        qf_values = qf_state.apply_fn(qf_state.params, observations)
        return jnp.argmax(qf_values, axis=1).reshape(-1)

    def _predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return DQNPolicy.select_action(self.qf_state, observation)
