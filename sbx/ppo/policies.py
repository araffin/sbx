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
from gym import spaces
from stable_baselines3.common.type_aliases import Schedule

# from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import BaseJaxPolicy

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class Critic(nn.Module):
    n_units: int = 256
    activation_fn: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_init: float = 0.0
    continuous: bool = True
    activation_fn: Callable = nn.tanh

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        mean = nn.Dense(self.action_dim)(x)
        # TODO: Allow state-independent exploration (default)
        # log_std = nn.Dense(self.action_dim)(x)
        # log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        # dist = TanhTransformedDistribution(
        #     tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        # )
        if self.continuous:
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        else:
            dist = tfd.Categorical(logits=mean)
        return dist


class PPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        ortho_init: bool = False,
        log_std_init: float = 0.0,
        activation_fn=nn.tanh,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
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
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        if net_arch is not None:
            assert isinstance(net_arch, list)
            self.n_units = net_arch[0]["pi"][0]
        else:
            self.n_units = 256
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key, lr_schedule: Schedule, max_grad_norm: float) -> None:
        key, actor_key, vf_key = jax.random.split(key, 3)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs = {
                "action_dim": np.prod(self.action_space.shape),
                "continuous": True,
            }
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": self.action_space.n,
                "continuous": False,
            }
        else:
            raise NotImplementedError(f"{self.action_space}")

        self.actor = Actor(
            n_units=self.n_units,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            **actor_kwargs,
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

        self.vf = Critic(n_units=self.n_units, activation_fn=self.activation_fn)

        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init({"params": vf_key}, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.inject_hyperparams(self.optimizer_class)(
                    learning_rate=lr_schedule(1),
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)
        self.vf.apply = jax.jit(self.vf.apply)

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

    def predict_all(self, observation: np.ndarray, key: jax.random.KeyArray) -> np.ndarray:
        return self._predict_all(self.actor, self.vf, self.actor_state, self.vf_state, observation, key)

    @staticmethod
    @partial(jax.jit, static_argnames=["actor", "vf"])
    def _predict_all(actor, vf, actor_state, vf_state, obervations, key):
        dist = actor.apply(actor_state.params, obervations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = vf.apply(vf_state.params, obervations).flatten()
        return actions, log_probs, values
