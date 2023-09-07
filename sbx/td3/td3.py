from functools import partial
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.td3.policies import TD3Policy


class TD3(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[TD3Policy]]] = {  # type: ignore[assignment]
        "MlpPolicy": TD3Policy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": TD3Policy,
    }

    policy: TD3Policy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=False,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            # pytype: disable=not-instantiable
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )
            # pytype: enable=not-instantiable

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {i: True for i in range(gradient_steps) if ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.key,
            (actor_loss_value, qf_loss_value),
        ) = self._train(
            self.gamma,
            self.tau,
            gradient_steps,
            data,
            policy_delay_indices,
            self.target_policy_noise,
            self.target_noise_clip,
            self.policy.qf_state,
            self.policy.actor_state,
            self.key,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())

    @staticmethod
    @jax.jit
    def update_critic(
        gamma: float,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        target_policy_noise: float,
        target_noise_clip: float,
        key: jax.random.KeyArray,
    ):
        key, noise_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)
        # Select action according to target net and add clipped noise
        next_state_actions = actor_state.apply_fn(actor_state.target_params, next_observations)
        noise = jax.random.normal(noise_key, actions.shape) * target_policy_noise
        noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
        next_state_actions = jnp.clip(next_state_actions + noise, -1.0, 1.0)

        #  Compute the next Q-values: min over all critics targets
        qf_next_values = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            next_state_actions,
            rngs={"dropout": dropout_key_target},
        )

        next_q_values = jnp.min(qf_next_values, axis=0)
        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values

        def mse_loss(params, dropout_key):
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf_state.apply_fn(params, observations, actions, rngs={"dropout": dropout_key})
            return 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()

        qf_loss_value, grads = jax.value_and_grad(mse_loss, has_aux=False)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            qf_loss_value,
            key,
        )

    @staticmethod
    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        observations: np.ndarray,
        key: jax.random.KeyArray,
    ):
        key, dropout_key = jax.random.split(key, 2)

        def actor_loss(params):
            actor_actions = actor_state.apply_fn(params, observations)

            qf_pi = qf_state.apply_fn(
                qf_state.params,
                observations,
                actor_actions,
                rngs={"dropout": dropout_key},
            )
            # Take min among all critics (mean for droq)
            min_qf_pi = jnp.min(qf_pi, axis=0)
            actor_loss = -min_qf_pi.mean()
            return actor_loss

        actor_loss_value, grads = jax.value_and_grad(actor_loss, has_aux=False)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState, actor_state: RLTrainState):
        qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, tau)
        )
        return qf_state, actor_state

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps"])
    def _train(
        cls,
        gamma: float,
        tau: float,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay_indices: flax.core.FrozenDict,
        target_policy_noise: float,
        target_noise_clip: float,
        qf_state: RLTrainState,
        actor_state: RLTrainState,
        key,
    ):
        actor_loss_value = jnp.array(0)

        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]

            (
                qf_state,
                qf_loss_value,
                key,
            ) = TD3.update_critic(
                gamma,
                actor_state,
                qf_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                target_policy_noise,
                target_noise_clip,
                key,
            )
            qf_state, actor_state = TD3.soft_update(tau, qf_state, actor_state)

            # hack to be able to jit (n_updates % policy_delay == 0)
            if i in policy_delay_indices:
                (actor_state, qf_state, actor_loss_value, key) = cls.update_actor(
                    actor_state,
                    qf_state,
                    slice(data.observations),
                    key,
                )

        return (
            qf_state,
            actor_state,
            key,
            (actor_loss_value, qf_loss_value),
        )
