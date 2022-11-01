from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from stable_baselines3.common.type_aliases import Schedule

# from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import BaseJaxPolicy

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class Critic(nn.Module):
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_init: float = 1.0
    # log_std_min: float = -20
    # log_std_max: float = 2

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        # TODO: Allow state-independent exploration (default)
        # log_std = nn.Dense(self.action_dim)(x)
        # log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        # dist = TanhTransformedDistribution(
        #     tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        # )
        log_std = self.param("log_std", constant(self.log_std_init), (1, self.action_dim))

        dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return dist


class PPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        # activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
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
            self.n_units = net_arch[0]
        else:
            self.n_units = 256
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key, lr_schedule: Schedule, max_grad_norm: float) -> None:
        key, actor_key, vf_key, dropout_key = jax.random.split(key, 4)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        self.actor = Actor(
            action_dim=np.prod(self.action_space.shape),
            n_units=self.n_units,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.inject_hyperparams(self.optimizer_class)(
                    learning_rate=lr_schedule(1),
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.vf = Critic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            n_units=self.n_units,
        )

        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init(
                {"params": vf_key, "dropout": dropout_key},
                obs,
            ),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.inject_hyperparams(self.optimizer_class)(
                    learning_rate=lr_schedule(1),
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.vf.apply = jax.jit(self.vf.apply, static_argnames=("dropout_rate", "use_layer_norm"))

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor, self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return BaseJaxPolicy.sample_action(self.actor, self.actor_state, observation, self.noise_key)

    def evaluate_action(self, observation: np.ndarray) -> np.ndarray:
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return self._evaluate_action(self.actor, self.vf, self.actor_state, self.vf_state, observation, self.noise_key)

    @staticmethod
    @partial(jax.jit, static_argnames=["actor", "vf"])
    def _evaluate_action(actor, vf, actor_state, vf_state, obervations, key):
        dist = actor.apply(actor_state.params, obervations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        values = vf.apply(
            vf_state.params,
            obervations,
            rngs={"dropout": key},
        ).flatten()
        return actions, log_probs, values
