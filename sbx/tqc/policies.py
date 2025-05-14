from typing import Any, Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import (
    BaseJaxPolicy,
    ContinuousCritic,
    GaussianActor,
    SimbaContinuousCritic,
    SimbaSquashedGaussianActor,
)
from sbx.common.type_aliases import RLTrainState


class TQCPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        top_quantiles_to_drop_per_net: int = 2,
        n_quantiles: int = 25,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        squash_output: bool = True,
        ortho_init: bool = False,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = GaussianActor,
        critic_class: type[nn.Module] = ContinuousCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        self.n_quantiles = n_quantiles
        self.n_critics = n_critics
        self.top_quantiles_to_drop_per_net = top_quantiles_to_drop_per_net
        # Sort and drop top k quantiles to control overestimation.
        quantiles_total = self.n_quantiles * self.n_critics
        top_quantiles_to_drop_per_net = self.top_quantiles_to_drop_per_net
        self.n_target_quantiles = quantiles_total - top_quantiles_to_drop_per_net * self.n_critics
        self.use_sde = use_sde
        self.activation_fn = activation_fn
        self.actor_class = actor_class
        self.critic_class = critic_class
        self.log_std_init = log_std_init
        self.ortho_init = ortho_init

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
        key, dropout_key1, dropout_key2, self.key = jax.random.split(key, 4)
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
            activation_fn=self.activation_fn,
            squash_output=self.squash_output,
            log_std_init=self.log_std_init,
            ortho_init=self.ortho_init,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # Inject hyperparameters to be able to modify it later
        # See https://stackoverflow.com/questions/78527164
        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(learning_rate=lr_schedule(1), **self.optimizer_kwargs)

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                # optax.clip_by_global_norm(max_grad_norm),
                optimizer_class,
            ),
        )

        self.qf = self.critic_class(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            output_dim=self.n_quantiles,
            activation_fn=self.activation_fn,
        )

        optimizer_class_qf = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=qf_learning_rate, **self.optimizer_kwargs
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
            tx=optax.chain(
                # optax.clip_by_global_norm(max_grad_norm),
                optimizer_class_qf,
            ),
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
            tx=optax.chain(
                # optax.clip_by_global_norm(max_grad_norm),
                optimizer_class_qf,
            ),
        )
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
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)


class SimbaTQCPolicy(TQCPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        dropout_rate: float = 0,
        layer_norm: bool = False,
        top_quantiles_to_drop_per_net: int = 2,
        n_quantiles: int = 25,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        squash_output: bool = True,
        ortho_init: bool = False,
        use_expln: bool = False,
        clip_mean: float = 2,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adamw,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = SimbaSquashedGaussianActor,
        critic_class: type[nn.Module] = SimbaContinuousCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            dropout_rate,
            layer_norm,
            top_quantiles_to_drop_per_net,
            n_quantiles,
            activation_fn,
            use_sde,
            log_std_init,
            squash_output,
            ortho_init,
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
            critic_class,
        )
