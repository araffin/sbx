# import copy
from collections import OrderedDict
from collections.abc import Sequence
from typing import Callable, Optional, Union, no_type_check

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.jax_layers import SimbaResidualBlock

tfd = tfp.distributions


class Flatten(nn.Module):
    """
    Equivalent to PyTorch nn.Flatten() layer.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))


class OneHot(nn.Module):
    """
    Convert int to one-hot representation.
    """

    num_classes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return Flatten()(jax.nn.one_hot(x, num_classes=self.num_classes))


class BaseJaxPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    @jax.jit
    def sample_action(actor_state, observations, key):
        dist = actor_state.apply_fn(actor_state.params, observations)
        action = dist.sample(seed=key)
        return action

    @staticmethod
    @jax.jit
    def select_action(actor_state, observations):
        return actor_state.apply_fn(actor_state.params, observations).mode()

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, dict[str, np.ndarray]]) -> tuple[np.ndarray, bool]:
        return prepare_obs(observation, self.observation_space)

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode


def prepare_obs(observation: Union[np.ndarray, dict[str, np.ndarray]], space: spaces.Space) -> tuple[np.ndarray, bool]:
    vectorized_env = False
    if isinstance(observation, dict):
        assert isinstance(space, spaces.Dict), f"The observation provided is a dict but the obs space is {space}"

        vectorized_env = is_vectorized_observation(observation, space)
        observation = jax.tree.map(
            lambda obs, obs_sp: prepare_obs(obs, obs_sp)[0], OrderedDict(observation), OrderedDict(space.spaces)
        )

    elif is_image_space(space):
        # Handle the different cases for images
        # as PyTorch use channel first format
        observation = maybe_transpose(observation, space)

    else:
        observation = np.array(observation)

    if not isinstance(space, spaces.Dict):
        assert isinstance(observation, np.ndarray)
        vectorized_env = is_vectorized_observation(observation, space)
        # Add batch dimension if needed
        observation = observation.reshape((-1, *space.shape))  # type: ignore[misc]

    return observation, vectorized_env


class ContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class SimbaContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False  # for consistency, not used
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1
    scale_factor: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        # Note: simba was using kernel_init=orthogonal_init(1)
        x = nn.Dense(self.net_arch[0])(x)
        for n_units in self.net_arch:
            x = SimbaResidualBlock(n_units, self.activation_fn, self.scale_factor)(x)
            # TODO: double check where to put the dropout
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        x = nn.LayerNorm()(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            ContinuousCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            output_dim=self.output_dim,
        )(obs, action)
        return q_values


class SimbaVectorCritic(nn.Module):
    net_arch: Sequence[int]
    # Note: we have use_layer_norm for consistency but it is not used (always on)
    use_layer_norm: bool = True
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            SimbaContinuousCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            output_dim=self.output_dim,
        )(obs, action)
        return q_values


class SquashedGaussianActor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class SimbaSquashedGaussianActor(nn.Module):
    # Note: each element in net_arch correpond to a residual block
    # not just a single layer
    net_arch: Sequence[int]
    action_dim: int
    # num_blocks: int = 2
    log_std_min: float = -20
    log_std_max: float = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)

        # Note: simba was using kernel_init=orthogonal_init(1)
        x = nn.Dense(self.net_arch[0])(x)
        for n_units in self.net_arch:
            x = SimbaResidualBlock(n_units, self.activation_fn, self.scale_factor)(x)
        x = nn.LayerNorm()(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist
