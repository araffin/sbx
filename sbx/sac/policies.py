from collections.abc import Callable
from typing import Any

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
    SimbaSquashedGaussianActor,
    SimbaVectorCritic,
    SquashedGaussianActor,
    VectorCritic,
)
from sbx.common.type_aliases import RLTrainState
from sbx.sample_dqn.policies import ENABLE_RERUN, SampleDQNPolicy


class SACPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = SquashedGaussianActor,
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
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
        else:
            self.net_arch_pi = self.net_arch_qf = [256, 256]
        self.n_critics = n_critics
        self.use_sde = use_sde
        self.activation_fn = activation_fn
        self.actor_class = actor_class
        self.vector_critic_class = vector_critic_class

        self.key = self.noise_key = jax.random.PRNGKey(0)
        self.n_steps = 0

    def build(self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float) -> jax.Array:
        key, actor_key, qf_key, dropout_key = jax.random.split(key, 4)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array([spaces.flatten(self.observation_space, self.observation_space.sample())])
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.action_dim = int(np.prod(self.action_space.shape))
        self.actor = self.actor_class(
            action_dim=int(np.prod(self.action_space.shape)),
            net_arch=self.net_arch_pi,
            activation_fn=self.activation_fn,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # Inject hyperparameters to be able to modify it later
        # See https://stackoverflow.com/questions/78527164
        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(learning_rate=lr_schedule(1), **self.optimizer_kwargs)

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optimizer_class,
        )

        self.qf = self.vector_critic_class(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,
            # No flatten layer because we repeat actions for sampling
            flatten=False,
        )

        optimizer_class_qf = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=qf_learning_rate, **self.optimizer_kwargs
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
            ),
            tx=optimizer_class_qf,
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
            action = BaseJaxPolicy.select_action(self.actor_state, observation)
        else:
            # Trick to use gSDE: repeat sampled noise by using the same noise key
            if not self.use_sde:
                self.reset_noise()
            action = BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)
        if ENABLE_RERUN:
            sampling_key, self.key = jax.random.split(self.key, 2)
            n_sampled_actions = 64
            n_iterations = 8
            n_top = 8
            extra_noise_std = 0.2774777566569773
            initial_variance = 0.6591638155238029
            cem_action = SampleDQNPolicy.select_action(
                self.qf_state,
                observation,
                sampling_key,
                # Increate search budget at test time
                n_sampled_actions,
                self.action_dim,
                # sampling_strategy=self.sampling_strategy.value,
                n_top=n_top,
                n_iterations=n_iterations,
                initial_variance=initial_variance,
                extra_noise_std=extra_noise_std,
            )
            from sbx.common.rerun_logging import log_step

            self.n_steps += 1
            # import ipdb; ipdb.set_trace()
            n_sampled_actions = 2
            repeated_obs = jnp.repeat(observation, n_sampled_actions, axis=0)
            combined_actions = jnp.concatenate([jnp.expand_dims(action[0], axis=1), jnp.expand_dims(cem_action[0], axis=1)])
            # shape = (n_critics, n_actions, 1)
            qf_values = self.qf_state.apply_fn(
                self.qf_state.params,
                repeated_obs,
                combined_actions,
                deterministic=True,  # disable dropout at inference
                rngs={"dropout": sampling_key},
            )
            # Only log first env
            log_step(self.n_steps, action[0], cem_action[0], qf_values)
            if deterministic:
                action = cem_action

        return action


class SimbaSACPolicy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        dropout_rate: float = 0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2,
        features_extractor_class=None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        # AdamW for simba
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adamw,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_class: type[nn.Module] = SimbaSquashedGaussianActor,
        vector_critic_class: type[nn.Module] = SimbaVectorCritic,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            dropout_rate,
            layer_norm,
            activation_fn,
            use_sde,
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
