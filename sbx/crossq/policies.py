from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.jax_layers import BatchRenorm, SimbaResidualBlock
from sbx.common.policies import BaseJaxPolicy, Flatten
from sbx.common.type_aliases import BatchNormTrainState

tfd = tfp.distributions


class Critic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    use_batch_norm: bool = True
    dropout_rate: Optional[float] = None
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        if self.use_batch_norm:
            x = BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warmup_steps=self.renorm_warmup_steps,
            )(x)
        else:
            # Create dummy batchstats
            BatchRenorm(use_running_average=not train)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
            if self.use_batch_norm:
                x = BatchRenorm(
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    warmup_steps=self.renorm_warmup_steps,
                )(x)

        x = nn.Dense(1)(x)
        return x


class SimbaCritic(nn.Module):
    net_arch: Sequence[int]
    dropout_rate: Optional[float] = None
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        norm_layer = partial(
            BatchRenorm,
            use_running_average=not train,
            momentum=self.batch_norm_momentum,
            warmup_steps=self.renorm_warmup_steps,
        )
        x = norm_layer()(x)
        x = nn.Dense(self.net_arch[0])(x)

        for n_units in self.net_arch:
            x = SimbaResidualBlock(
                n_units,
                self.activation_fn,
                self.scale_factor,
                norm_layer,  # type: ignore[arg-type]
            )(x)
            # TODO: double check where to put the dropout
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        x = norm_layer()(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = False):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True, "batch_stats": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            renorm_warmup_steps=self.renorm_warmup_steps,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(obs, action, train)
        return q_values


class SimbaVectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False  # ignored
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, train: bool = False):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            SimbaCritic,
            variable_axes={"params": 0, "batch_stats": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True, "batch_stats": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            # use_layer_norm=self.use_layer_norm,
            # use_batch_norm=self.use_batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            renorm_warmup_steps=self.renorm_warmup_steps,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            scale_factor=self.scale_factor,
        )(obs, action, train)
        return q_values


class SimbaActor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)
        norm_layer = partial(
            BatchRenorm,
            use_running_average=not train,
            momentum=self.batch_norm_momentum,
            warmup_steps=self.renorm_warmup_steps,
        )
        x = norm_layer()(x)
        x = nn.Dense(self.net_arch[0])(x)

        for n_units in self.net_arch:
            x = SimbaResidualBlock(
                n_units,
                self.activation_fn,
                self.scale_factor,
                norm_layer,  # type: ignore[arg-type]
            )(x)
        x = norm_layer()(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class Actor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    log_std_min: float = -20
    log_std_max: float = 2
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.99
    renorm_warmup_steps: int = 100_000
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)
        if self.use_batch_norm:
            x = BatchRenorm(
                use_running_average=not train,
                momentum=self.batch_norm_momentum,
                warmup_steps=self.renorm_warmup_steps,
            )(x)
        else:
            # Create dummy batchstats
            BatchRenorm(use_running_average=not train)(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
            if self.use_batch_norm:
                x = BatchRenorm(
                    use_running_average=not train,
                    momentum=self.batch_norm_momentum,
                    warmup_steps=self.renorm_warmup_steps,
                )(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class CrossQPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        batch_norm: bool = True,  # for critic
        batch_norm_actor: bool = True,
        batch_norm_momentum: float = 0.99,
        renorm_warmup_steps: int = 100_000,
        use_sde: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
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
        actor_class: Type[nn.Module] = Actor,
        vector_critic_class: Type[nn.Module] = VectorCritic,
    ):
        if optimizer_kwargs is None:
            # Note: the default value for b1 is 0.9 in Adam.
            # b1=0.5 is used in the original CrossQ implementation
            # but shows only little overall improvement.
            optimizer_kwargs = {}
            if optimizer_class in [optax.adam, optax.adamw]:
                optimizer_kwargs["b1"] = 0.5

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
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_actor = batch_norm_actor
        self.renorm_warmup_steps = renorm_warmup_steps
        self.actor_class = actor_class
        self.vector_critic_class = vector_critic_class

        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = [256, 256]
            # While CrossQ already works with a [256,256] critic network,
            # the authors found that a wider network significantly improves performance.
            # We use a slightly smaller net for faster computation, [1024, 1024] instead of [2048, 2048] in the paper
            self.net_arch_qf = [1024, 1024]

        self.n_critics = n_critics
        self.use_sde = use_sde
        self.activation_fn = activation_fn

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.actor = self.actor_class(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            use_batch_norm=self.batch_norm_actor,
            batch_norm_momentum=self.batch_norm_momentum,
            renorm_warmup_steps=self.renorm_warmup_steps,
            activation_fn=self.activation_fn,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # Note: re-use same bn_key as for the critic?
        actor_params = self.actor.init(
            {"params": actor_key, "batch_stats": bn_key},
            obs,
            train=False,
        )

        self.actor_state = BatchNormTrainState.create(
            apply_fn=self.actor.apply,
            params=actor_params["params"],
            batch_stats=actor_params["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.qf = self.vector_critic_class(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            use_batch_norm=self.batch_norm,
            batch_norm_momentum=self.batch_norm_momentum,
            renorm_warmup_steps=self.renorm_warmup_steps,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,
        )

        qf_params = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )

        self.qf_state = BatchNormTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_params["params"],
            batch_stats=qf_params["batch_stats"],
            tx=self.optimizer_class(
                learning_rate=qf_learning_rate,  # type: ignore[call-arg]
                **self.optimizer_kwargs,
            ),
        )

        self.actor.apply = jax.jit(  # type: ignore[method-assign]
            self.actor.apply,
            static_argnames=("use_batch_norm"),
        )
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm", "use_batch_norm"),
        )

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    @staticmethod
    @jax.jit
    def sample_action(actor_state, obervations, key):
        dist = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
            obervations,
            train=False,
        )
        action = dist.sample(seed=key)
        return action

    @staticmethod
    @jax.jit
    def select_action(actor_state, obervations):
        return actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
            obervations,
            train=False,
        ).mode()

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:  # type: ignore[override]
        if deterministic:
            return self.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return self.sample_action(self.actor_state, observation, self.noise_key)


class SimbaCrossQPolicy(CrossQPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0,
        layer_norm: bool = False,
        batch_norm: bool = True,
        batch_norm_actor: bool = True,
        batch_norm_momentum: float = 0.99,
        renorm_warmup_steps: int = 100000,
        use_sde: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: Type[nn.Module] = SimbaActor,  # TODO: replace with Simba actor
        vector_critic_class: Type[nn.Module] = SimbaVectorCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            dropout_rate,
            layer_norm,
            batch_norm,
            batch_norm_actor,
            batch_norm_momentum,
            renorm_warmup_steps,
            use_sde,
            activation_fn,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
            actor_class,
            vector_critic_class,
        )
