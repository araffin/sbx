from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import BaseJaxPolicy
from sbx.common.type_aliases import RLTrainState

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


@jax.jit
def avg_l1_norm(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / jnp.clip(jnp.mean(jnp.abs(x), axis=-1, keepdims=True), a_min=eps)


class StateEncoder(nn.Module):
    net_arch: Sequence[int]
    embedding_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = nn.elu(x)
        x = nn.Dense(self.embedding_dim)(x)
        return avg_l1_norm(x)


class StateActionEncoder(nn.Module):
    net_arch: Sequence[int]
    embedding_dim: int = 256

    @nn.compact
    def __call__(self, z_state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([z_state, action], -1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = nn.elu(x)
        x = nn.Dense(self.embedding_dim)(x)
        return avg_l1_norm(x)


class Critic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, z_state: jnp.ndarray, z_state_action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, action], -1)
        embeddings = jnp.concatenate([z_state, z_state_action], -1)
        x = avg_l1_norm(nn.Dense(self.net_arch[0])(x))
        # Combine with embeddings
        x = jnp.concatenate([x, embeddings], -1)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, z_state: jnp.ndarray, z_state_action: jnp.ndarray):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
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
        )(obs, action, z_state, z_state_action)
        return q_values


class Actor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, z_state: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = avg_l1_norm(nn.Dense(self.net_arch[0])(x))
        # Combine with encoding
        x = jnp.concatenate([x, z_state], -1)
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class SAC7Policy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        embedding_dim: int = 256,
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
        n_critics: int = 2,
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
        self.embedding_dim = embedding_dim
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
                self.net_arch_encoder = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
                self.net_arch_encoder = net_arch["encoder"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
            self.net_arch_encoder = [256, 256]
        self.n_critics = n_critics
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.random.KeyArray, lr_schedule: Schedule, qf_learning_rate: float) -> jax.random.KeyArray:
        key, actor_key, qf_key, dropout_key = jax.random.split(key, 4)
        # Keys for the encoder and state action encoder
        key, encoder_key, action_encoder_key = jax.random.split(key, 3)

        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        z_state = jnp.zeros((1, self.embedding_dim))
        z_state_action = jnp.zeros((1, self.embedding_dim))

        self.actor = Actor(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs, z_state),
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf = VectorCritic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
                z_state,
                z_state_action,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
                z_state,
                z_state_action,
            ),
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.encoder = StateEncoder(
            net_arch=self.net_arch_encoder,
            embedding_dim=self.embedding_dim,
        )
        self.action_encoder = StateActionEncoder(
            net_arch=self.net_arch_encoder,
            embedding_dim=self.embedding_dim,
        )

        self.encoder_state = RLTrainState.create(
            apply_fn=self.encoder.apply,
            params=self.encoder.init(
                {"params": encoder_key},
                obs,
            ),
            target_params=self.encoder.init(
                {"params": encoder_key},
                obs,
            ),
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.action_encoder_state = RLTrainState.create(
            apply_fn=self.action_encoder.apply,
            params=self.action_encoder.init(
                {"params": action_encoder_key},
                z_state,
                action,
            ),
            target_params=self.action_encoder.init(
                {"params": action_encoder_key},
                z_state,
                action,
            ),
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )
        self.encoder.apply = jax.jit(self.encoder.apply)  # type: ignore[method-assign]
        self.action_encoder.apply = jax.jit(self.action_encoder.apply)  # type: ignore[method-assign]
        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )

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
            return SAC7Policy.select_action(self.actor_state, self.encoder_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return SAC7Policy.sample_action(self.actor_state, self.encoder_state, observation, self.noise_key)

    @staticmethod
    @jax.jit
    def sample_action(actor_state, encoder_state, obervations, key):
        z_state = encoder_state.apply_fn(encoder_state.params, obervations)
        dist = actor_state.apply_fn(actor_state.params, obervations, z_state)
        action = dist.sample(seed=key)
        return action

    @staticmethod
    @jax.jit
    def select_action(actor_state, encoder_state, obervations):
        z_state = encoder_state.apply_fn(encoder_state.params, obervations)
        return actor_state.apply_fn(actor_state.params, obervations, z_state).mode()
