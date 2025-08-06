from functools import partial
from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import BaseJaxPolicy, ContinuousCritic
from sbx.common.type_aliases import RLTrainState


class SampleDQNPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_sampled_actions: int = 10,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is not None:
            assert isinstance(net_arch, list)
            self.net_arch = net_arch
        else:
            self.net_arch = [256, 256]

        self.activation_fn = activation_fn
        self.n_sampled_actions = n_sampled_actions
        self.action_dim = int(np.prod(self.action_space.shape))

    def build(self, key: jax.Array, lr_schedule: Schedule) -> jax.Array:
        key, qf_key = jax.random.split(key, 2)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize sampling
        self.update_sampling_key()

        obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.qf: nn.Module = ContinuousCritic(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            flatten=False,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs, action),
            target_params=self.qf.init({"params": qf_key}, obs, action),
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
    @partial(jax.jit, static_argnames=["n_sampled_actions", "action_dim"])
    def select_action(qf_state, observations, key, n_sampled_actions: int, action_dim: int):
        # Uniform sampling
        # actions = jax.random.uniform(
        #     key,
        #     shape=(observations.shape[0], n_sampled_actions, action_dim),
        #     minval=-1.0,
        #     maxval=1.0,
        # )
        # Gaussian dist
        scale = 1.0
        actions = scale * jax.random.normal(key, shape=(observations.shape[0], n_sampled_actions, action_dim))
        actions = jnp.clip(actions, -1.0, 1.0)

        repeated_obs = jnp.repeat(jnp.expand_dims(observations, axis=1), n_sampled_actions, axis=1)
        qf_values = qf_state.apply_fn(qf_state.params, repeated_obs, actions)

        actions_indices = jnp.argmax(qf_values, axis=1)

        indices_expanded = jnp.expand_dims(actions_indices, axis=-1)  # shape (batch_size, 1, 1)
        best_actions = jnp.take_along_axis(actions, indices_expanded, axis=1)  # shape (batch_size, 1, action_dim)
        best_actions = best_actions.squeeze(axis=1)  # shape (batch_size, action_dim)
        return best_actions

    def update_sampling_key(self) -> None:
        self.key, self.sampling_key = jax.random.split(self.key, 2)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:  # type: ignore[override]
        # Note: deterministic is currently not properly handled
        self.update_sampling_key()
        return SampleDQNPolicy.select_action(
            self.qf_state,
            observation,
            self.sampling_key,
            # Increate search budget at test time
            2 * self.n_sampled_actions if deterministic else self.n_sampled_actions,
            self.action_dim,
        )
        # if deterministic:
        #     return SampleDQNPolicy.select_action(self.qf_state, observation, self.sampling_key)
        # return SampleDQNPolicy.sample_action(self.qf_state, observation, self.sampling_key)
