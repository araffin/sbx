from collections.abc import Sequence
from dataclasses import field
from typing import Any, Callable, Optional, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.jax_layers import SimbaResidualBlock
from sbx.common.policies import BaseJaxPolicy, Flatten

tfd = tfp.distributions


class Critic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)

        x = nn.Dense(1)(x)
        return x


class SimbaCritic(nn.Module):
    net_arch: Sequence[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = nn.Dense(self.net_arch[0])(x)
        for n_units in self.net_arch:
            x = SimbaResidualBlock(n_units, self.activation_fn, self.scale_factor)(x)

        x = nn.LayerNorm()(x)
        x = nn.Dense(1)(x)
        return x


class Actor(nn.Module):
    action_dim: int
    net_arch: Sequence[int]
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    # For Discrete, MultiDiscrete and MultiBinary actions
    num_discrete_choices: Optional[Union[int, Sequence[int]]] = None
    # For MultiDiscrete
    max_num_choices: int = 0
    split_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    # Last layer with small scale
    ortho_init: bool = False

    def get_std(self) -> jnp.ndarray:
        # Make it work with gSDE
        return jnp.array(0.0)

    def __post_init__(self) -> None:
        # For MultiDiscrete
        if isinstance(self.num_discrete_choices, np.ndarray):
            self.max_num_choices = max(self.num_discrete_choices)
            # np.cumsum(...) gives the correct indices at which to split the flatten logits
            self.split_indices = np.cumsum(self.num_discrete_choices[:-1])
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)

        if self.ortho_init:
            orthogonal_init = nn.initializers.orthogonal(scale=0.01)
            bias_init = nn.initializers.zeros
            action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal_init, bias_init=bias_init)(x)

        else:
            action_logits = nn.Dense(self.action_dim)(x)

        log_std = jnp.zeros(1)
        if self.num_discrete_choices is None:
            # Continuous actions
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))
        elif isinstance(self.num_discrete_choices, int):
            dist = tfd.Categorical(logits=action_logits)
        else:
            # Split action_logits = (batch_size, total_choices=sum(self.num_discrete_choices))
            action_logits = jnp.split(action_logits, self.split_indices, axis=1)
            # Pad to the maximum number of choices (required by tfp.distributions.Categorical).
            # Pad by -inf, so that the probability of these invalid actions is 0.
            logits_padded = jnp.stack(
                [
                    jnp.pad(
                        logit,
                        # logit is of shape (batch_size, n)
                        # only pad after dim=1, to max_num_choices - n
                        # pad_width=((before_dim_0, after_0), (before_dim_1, after_1))
                        pad_width=((0, 0), (0, self.max_num_choices - logit.shape[1])),
                        constant_values=-np.inf,
                    )
                    for logit in action_logits
                ],
                axis=1,
            )
            dist = tfp.distributions.Independent(
                tfp.distributions.Categorical(logits=logits_padded), reinterpreted_batch_ndims=1
            )
        return dist


class SimbaActor(nn.Module):
    action_dim: int
    net_arch: Sequence[int]
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    # For Discrete, MultiDiscrete and MultiBinary actions
    num_discrete_choices: Optional[Union[int, Sequence[int]]] = None
    # For MultiDiscrete
    max_num_choices: int = 0
    # Last layer with small scale
    ortho_init: bool = False
    scale_factor: int = 4

    def get_std(self) -> jnp.ndarray:
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)

        x = nn.Dense(self.net_arch[0])(x)
        for n_units in self.net_arch:
            x = SimbaResidualBlock(n_units, self.activation_fn, self.scale_factor)(x)
        x = nn.LayerNorm()(x)

        if self.ortho_init:
            orthogonal_init = nn.initializers.orthogonal(scale=0.01)
            bias_init = nn.initializers.zeros
            mean_action = nn.Dense(self.action_dim, kernel_init=orthogonal_init, bias_init=bias_init)(x)

        else:
            mean_action = nn.Dense(self.action_dim)(x)

        # Continuous actions
        log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
        dist = tfd.MultivariateNormalDiag(loc=mean_action, scale_diag=jnp.exp(log_std))

        return dist


class PPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        ortho_init: bool = False,
        log_std_init: float = 0.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = Actor,
        critic_class: type[nn.Module] = Critic,
    ):
        if optimizer_kwargs is None:
            # Small values to avoid NaN in Adam optimizer
            optimizer_kwargs = {}
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_vf = net_arch
            else:
                assert isinstance(net_arch, dict)
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_vf = net_arch["vf"]
        else:
            self.net_arch_pi = self.net_arch_vf = [64, 64]
        self.use_sde = use_sde
        self.ortho_init = ortho_init
        self.actor_class = actor_class
        self.critic_class = critic_class

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, max_grad_norm: float) -> jax.Array:
        key, actor_key, vf_key = jax.random.split(key, 3)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs = {
                "action_dim": int(np.prod(self.action_space.shape)),
            }
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            }
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            assert self.action_space.nvec.ndim == 1, (
                f"Only one-dimensional MultiDiscrete action spaces are supported, "
                f"but found MultiDiscrete({(self.action_space.nvec).tolist()})."
            )
            actor_kwargs = {
                "action_dim": int(np.sum(self.action_space.nvec)),
                "num_discrete_choices": self.action_space.nvec,  # type: ignore[dict-item]
            }
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            # Handle binary action spaces as discrete action spaces with two choices.
            actor_kwargs = {
                "action_dim": 2 * self.action_space.n,
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            }
        else:
            raise NotImplementedError(f"{self.action_space}")

        self.actor = self.actor_class(
            net_arch=self.net_arch_pi,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            ortho_init=self.ortho_init,
            **actor_kwargs,  # type: ignore[arg-type]
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # Inject hyperparameters to be able to modify it later
        # See https://stackoverflow.com/questions/78527164
        # Note: eps=1e-5 for Adam
        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(learning_rate=lr_schedule(1), **self.optimizer_kwargs)

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer_class,
            ),
        )

        self.vf = self.critic_class(net_arch=self.net_arch_vf, activation_fn=self.activation_fn)

        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init({"params": vf_key}, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optimizer_class,
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:  # type: ignore[override]
        if deterministic:
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)

    def predict_all(self, observation: np.ndarray, key: jax.Array) -> np.ndarray:
        return self._predict_all(self.actor_state, self.vf_state, observation, key)

    @staticmethod
    @jax.jit
    def _predict_all(actor_state, vf_state, observations, key):
        dist = actor_state.apply_fn(actor_state.params, observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = vf_state.apply_fn(vf_state.params, observations).flatten()
        return actions, log_probs, values


class SimbaPPOPolicy(PPOPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        ortho_init: bool = False,
        log_std_init: float = 0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        use_sde: bool = False,
        use_expln: bool = False,
        clip_mean: float = 2,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = SimbaActor,
        critic_class: type[nn.Module] = SimbaCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            ortho_init,
            log_std_init,
            activation_fn,
            use_sde,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            share_features_extractor,
            actor_class,
            critic_class,
        )
