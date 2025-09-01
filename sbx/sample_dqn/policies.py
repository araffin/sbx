from enum import Enum
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


class SamplingStrategy(Enum):
    UNIFORM = 0
    GAUSSIAN = 1
    CEM = 1


NAME_TO_SAMPLING_STRATEGY = {
    "uniform": SamplingStrategy.UNIFORM,
    "gaussian": SamplingStrategy.GAUSSIAN,
    "cem": SamplingStrategy.CEM,
}


@partial(jax.jit, static_argnames=["n_sampled_actions", "action_dim"])
def find_best_actions_cem(
    qf_state,
    observations,
    key,
    n_sampled_actions: int,
    action_dim: int,
    n_top: int = 6,
    n_iterations: int = 10,
    initial_variance: float = 1.0**2,
    extra_noise_std: float = 0.1,
):
    """
    Noisy Cross Entropy Method: http://dx.doi.org/10.1162/neco.2006.18.12.2936
    "Learning Tetris Using the Noisy Cross-Entropy Method"

    See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/62
    """
    best_actions = jnp.zeros((observations.shape[0], action_dim))
    best_actions_cov = jnp.ones_like(best_actions) * initial_variance
    extra_variance = jnp.ones_like(best_actions_cov) * extra_noise_std**2
    # Decay the extra noise in half the iterations
    extra_var_decay_time = n_iterations / 2.0

    carry = {
        "best_actions": best_actions,
        "best_actions_cov": best_actions_cov,
        "top_one_actions": best_actions,
        "key": key,
    }

    def one_update(i: int, carry: dict[str, Any]) -> dict[str, Any]:
        best_actions = carry["best_actions"]
        best_actions_cov = carry["best_actions_cov"]
        key = carry["key"]
        key, new_key = jax.random.split(key, 2)
        # Reduce extra variance over time
        extra_var_multiplier = jnp.max(jnp.array([(1.0 - i / extra_var_decay_time), 0]))
        # Sample using only the diagonal of the covariance matrix (+ extra noise)
        # TODO: try with full covariance?
        deltas = jax.random.normal(key, shape=(observations.shape[0], n_sampled_actions, action_dim))
        actions = jnp.expand_dims(best_actions, axis=1) + deltas * jnp.expand_dims(
            jnp.sqrt(best_actions_cov + extra_variance * extra_var_multiplier), axis=1
        )
        # actions = jnp.clip(actions, -1.0, 1.0)

        repeated_obs = jnp.repeat(jnp.expand_dims(observations, axis=1), n_sampled_actions, axis=1)
        # Shape is (n_critics, batch_size, n_repeated_actions, 1)
        qf_values = qf_state.apply_fn(qf_state.params, repeated_obs, jnp.clip(actions, -1.0, 1.0))
        # Twin network: take the min between q-networks
        # qf_values = jnp.min(qf_values, axis=0)
        # More optimistic alternative
        qf_values = jnp.mean(qf_values, axis=0)

        # Keep only the top performing candidates for update
        # Shape is (batch_size, n_top, 1)
        actions_indices = jnp.argsort(qf_values, axis=1, descending=True)[:, :n_top, :]
        # Shape (batch_size, n_top, action_dim)
        best_actions = jnp.take_along_axis(actions, actions_indices, axis=1)

        # Update centroid: barycenter of the best candidates
        return {
            "top_one_actions": best_actions[:, :1, :].squeeze(axis=1),
            "best_actions": best_actions.mean(axis=1),
            "best_actions_cov": best_actions.var(axis=1),
            "key": new_key,
        }

    update_carry = jax.lax.fori_loop(0, n_iterations, one_update, carry)
    best_actions = update_carry["best_actions"]
    # best_actions = update_carry["top_one_actions"]

    # shape (batch_size, action_dim)
    # return best_actions
    return jnp.clip(best_actions, -1.0, 1.0)


@partial(jax.jit, static_argnames=["n_sampled_actions", "action_dim", "gaussian_dist"])
def find_best_actions_sample_dist(
    qf_state,
    observations,
    key,
    n_sampled_actions: int,
    action_dim: int,
    gaussian_dist: bool = True,
):
    # Gaussian distribution
    if gaussian_dist:
        scale = 1.0
        actions = scale * jax.random.normal(key, shape=(observations.shape[0], n_sampled_actions, action_dim))
        actions = jnp.clip(actions, -1.0, 1.0)
        # Note: we could also use tanh, but doesn't work
        # actions = jnp.tanh(actions)
    else:
        # Uniform sampling
        actions = jax.random.uniform(
            key,
            shape=(observations.shape[0], n_sampled_actions, action_dim),
            minval=-1.0,
            maxval=1.0,
        )

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
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        n_sampled_actions: int = 100,
        sampling_strategy: Union[str, SamplingStrategy] = SamplingStrategy.CEM,
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
        if isinstance(sampling_strategy, str):
            sampling_strategy = NAME_TO_SAMPLING_STRATEGY[sampling_strategy]
        self.sampling_strategy = sampling_strategy

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
    def select_action(
        qf_state,
        observations,
        key,
        n_sampled_actions: int,
        action_dim: int,
        sampling_strategy: int = SamplingStrategy.CEM.value,
    ):
        return jax.lax.cond(
            sampling_strategy == SamplingStrategy.CEM.value,
            # If True: CEM
            partial(find_best_actions_cem, n_sampled_actions=n_sampled_actions, action_dim=action_dim),
            # If False: Gaussian/Uniform sampling
            partial(find_best_actions_sample_dist, n_sampled_actions=n_sampled_actions, action_dim=action_dim),
            qf_state,
            observations,
            key,
        )

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
            sampling_strategy=self.sampling_strategy.value,
        )
        # if deterministic:
        #     return SampleDQNPolicy.select_action(self.qf_state, observation, self.sampling_key)
        # return SampleDQNPolicy.sample_action(self.qf_state, observation, self.sampling_key)
