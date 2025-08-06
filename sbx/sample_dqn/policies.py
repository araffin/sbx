from functools import partial
from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import BaseJaxPolicy, VectorCritic
from sbx.common.type_aliases import RLTrainState


class SampleDQNPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        n_sampled_actions: int = 100,
        vector_critic_class: type[nn.Module] = VectorCritic,
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
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        if net_arch is not None:
            assert isinstance(net_arch, list)
            self.net_arch = net_arch
        else:
            self.net_arch = [256, 256]

        self.activation_fn = activation_fn
        self.n_sampled_actions = n_sampled_actions
        self.action_dim = int(np.prod(self.action_space.shape))
        self.n_critics = n_critics
        self.vector_critic_class = vector_critic_class
        self.use_sde = use_sde

    def build(self, key: jax.Array, lr_schedule: Schedule) -> jax.Array:
        key, qf_key = jax.random.split(key, 2)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize sampling
        self.update_sampling_key()

        obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.qf = self.vector_critic_class(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch,
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,
            # No flatten layer because we repeat actions for sampling
            flatten=False,
        )

        # TODO: enable dropout?
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init({"params": qf_key}, obs, action),
            target_params=self.qf.init({"params": qf_key}, obs, action),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )
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
        # Note: we could also use tanh
        actions = jnp.clip(actions, -1.0, 1.0)

        repeated_obs = jnp.repeat(jnp.expand_dims(observations, axis=1), n_sampled_actions, axis=1)
        # Shape is (n_critics, batch_size, n_repeated_actions, 1)
        qf_values = qf_state.apply_fn(qf_state.params, repeated_obs, actions)
        # Twin network: take the min between q-networks
        qf_values = jnp.min(qf_values, axis=0)

        actions_indices = jnp.argmax(qf_values, axis=1)

        indices_expanded = jnp.expand_dims(actions_indices, axis=-1)  # shape (batch_size, 1, 1)
        best_actions = jnp.take_along_axis(actions, indices_expanded, axis=1)  # shape (batch_size, 1, action_dim)
        best_actions = best_actions.squeeze(axis=1)  # shape (batch_size, action_dim)
        return best_actions

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        For interface compatibility when using gSDE.
        """
        self.update_sampling_key()

    def update_sampling_key(self) -> None:
        self.key, self.sampling_key = jax.random.split(self.key, 2)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:  # type: ignore[override]
        # Note: deterministic is currently not properly handled
        # Trick to use gSDE: repeat sampled actions by using the same noise key
        if not self.use_sde:
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
