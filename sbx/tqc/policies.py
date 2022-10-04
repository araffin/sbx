import copy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import is_vectorized_observation

from sbx.common.distributions import StateDependentNoiseDistribution, TanhTransformedDistribution
from sbx.common.type_aliases import RLTrainState

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


@partial(jax.jit, static_argnames="actor")
def sample_action(actor, actor_state, obervations, key):
    dist = actor.apply(actor_state.params, obervations)
    action = dist.sample(seed=key)
    return action


@partial(jax.jit, static_argnames="actor")
def select_action(actor, actor_state, obervations):
    return actor.apply(actor_state.params, obervations).mode()


class Critic(nn.Module):
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_quantiles: int = 25
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.n_units)(x)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_quantiles)(x)
        return x


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        # dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class ActorSDE(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        # gSDE
        latent_sde = x
        log_std = self.param("log_std", constant(-3), (latent_sde.shape[1], self.action_dim))
        # TODO: use expln trick instead
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)
        variance = (jax.lax.stop_gradient(latent_sde) ** 2 @ std**2) + 1e-6

        mean = nn.Dense(self.action_dim)(x)
        # hard_tanh that clips at [-2, 2]
        mean = jnp.where(mean > 2.0, 2.0, jnp.where(mean < -2.0, -2.0, mean))
        # dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = TanhTransformedDistribution(
            StateDependentNoiseDistribution(loc=mean, scale_diag=jnp.sqrt(variance), std=std),
        )
        return dist


class TQCPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        top_quantiles_to_drop_per_net: int = 2,
        n_quantiles: int = 25,
        # activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        # share_features_extractor: bool = False,
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
        self.n_units = 256
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        self.use_sde = True
        # Sort and drop top k quantiles to control overestimation.
        quantiles_total = self.n_quantiles * self.n_critics
        top_quantiles_to_drop_per_net = self.top_quantiles_to_drop_per_net
        self.n_target_quantiles = quantiles_total - top_quantiles_to_drop_per_net * self.n_critics

        self.key = jax.random.PRNGKey(0)

    def build(self, key, lr_schedule: Schedule) -> None:
        key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
        key, dropout_key1, dropout_key2, self.key = jax.random.split(key, 4)

        obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        if self.use_sde:
            self.actor = ActorSDE(
                action_dim=np.prod(self.action_space.shape),
                n_units=self.n_units,
            )
        else:
            self.actor = Actor(
                action_dim=np.prod(self.action_space.shape),
                n_units=self.n_units,
            )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=self.optimizer_class(learning_rate=lr_schedule(1), **self.optimizer_kwargs),
        )

        self.qf = Critic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            n_units=self.n_units,
            n_quantiles=self.n_quantiles,
        )

        self.qf1_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf1_key, "dropout": dropout_key1},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf1_key, "dropout": dropout_key1},
                obs,
                action,
            ),
            tx=optax.adam(learning_rate=lr_schedule(1)),
        )
        self.qf2_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf2_key, "dropout": dropout_key2},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf2_key, "dropout": dropout_key2},
                obs,
                action,
            ),
            tx=self.optimizer_class(learning_rate=lr_schedule(1), **self.optimizer_kwargs),
        )
        self.actor.apply = jax.jit(self.actor.apply)
        self.qf.apply = jax.jit(self.qf.apply, static_argnames=("dropout_rate", "use_layer_norm"))

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return select_action(self.actor, self.actor_state, observation)
        self.key, noise_key = jax.random.split(self.key, 2)
        return sample_action(self.actor, self.actor_state, observation, noise_key)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1,) + self.action_space.shape)

        if isinstance(self.action_space, gym.spaces.Box):
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
            actions = actions.squeeze(axis=0)

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode
