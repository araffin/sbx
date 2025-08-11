from functools import partial
from typing import Any, ClassVar, Optional, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.sample_dqn.policies import SampleDQNPolicy


class SampleDQN(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[dict[str, type[SampleDQNPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SampleDQNPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    policy: SampleDQNPolicy

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0.99,
        n_sampled_actions: int = 100,
        action_noise: Optional[ActionNoise] = None,
        # target_update_interval: int = 1000,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        # max_grad_norm: float = 10,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
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
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(gym.spaces.Box,),
            support_multi_env=True,
        )

        self.n_sampled_actions = n_sampled_actions
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                n_sampled_actions=self.n_sampled_actions,
                **self.policy_kwargs,
            )

            self.key = self.policy.build(self.key, self.lr_schedule)
            self.qf = self.policy.qf

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SampleDQN",
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

        if data.discounts is None:
            discounts = np.full((batch_size * gradient_steps,), self.gamma, dtype=np.float32)
        else:
            # For bootstrapping with n-step returns
            discounts = data.discounts.numpy().flatten()
        # Convert to numpy
        data = ReplayBufferSamplesNp(
            data.observations.numpy(),
            # Convert to int64
            data.actions.long().numpy(),
            data.next_observations.numpy(),
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
            discounts,
        )
        # Pre compute the slice indices
        # otherwise jax will complain
        indices = jnp.arange(len(data.dones)).reshape(gradient_steps, batch_size)

        update_carry = {
            "key": self.key,
            # To give the right shape and avoid JIT errors
            "sampled_actions": jnp.ones(self.n_sampled_actions),
            "tau": self.tau,
            "qf_state": self.policy.qf_state,
            "data": data,
            "indices": indices,
            "info": {
                "critic_loss": jnp.array([0.0]),
                "qf_mean_value": jnp.array([0.0]),
            },
        }

        # jit the loop similar to https://github.com/Howuhh/sac-n-jax
        # we use scan to be able to play with unroll parameter
        update_carry, _ = jax.lax.scan(
            self._train,
            update_carry,
            indices,
            unroll=1,
        )

        self.policy.qf_state = update_carry["qf_state"]
        self.key = update_carry["key"]
        qf_loss_value = update_carry["info"]["critic_loss"]
        qf_mean_value = update_carry["info"]["qf_mean_value"] / gradient_steps

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/qf_mean_value", qf_mean_value.item())

    @staticmethod
    @partial(jax.jit, static_argnames=["n_sampled_actions", "action_dim"])
    def find_max_target_q_cem(
        qf_state,
        observations,
        key,
        n_sampled_actions: int,
        action_dim: int,
        n_top: int = 5,
        n_iterations: int = 2,
    ):
        """
        Noisy Cross Entropy Method: http://dx.doi.org/10.1162/neco.2006.18.12.2936
        "Learning Tetris Using the Noisy Cross-Entropy Method"

        See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/62
        """
        initial_variance = 1.0**2
        extra_noise_std = 0.01
        best_actions = jnp.zeros((observations.shape[0], action_dim))
        best_actions_cov = jnp.ones_like(best_actions) * initial_variance
        extra_variance = jnp.ones_like(best_actions_cov) * extra_noise_std**2

        carry = {
            "best_actions": best_actions,
            "best_actions_cov": best_actions_cov,
            "next_q_values": jnp.zeros((observations.shape[0], 1)),
        }

        def one_update(i: int, carry: dict[str, Any]) -> dict[str, Any]:
            best_actions = carry["best_actions"]
            best_actions_cov = carry["best_actions_cov"]

            # Sample using only the diagonal of the covariance matrix (+ extra noise)
            # TODO: try with full covariance?
            deltas = jax.random.normal(key, shape=(observations.shape[0], n_sampled_actions, action_dim))
            actions = jnp.expand_dims(best_actions, axis=1) + deltas * jnp.expand_dims(
                jnp.sqrt(best_actions_cov + extra_variance), axis=1
            )
            actions = jnp.clip(actions, -1.0, 1.0)

            repeated_obs = jnp.repeat(jnp.expand_dims(observations, axis=1), n_sampled_actions, axis=1)
            # Shape is (n_critics, batch_size, n_repeated_actions, 1)
            qf_next_values = qf_state.apply_fn(qf_state.target_params, repeated_obs, actions)
            # Twin network: take the min between q-networks
            qf_next_values = jnp.min(qf_next_values, axis=0)

            # Keep only the top performing candidates for update
            # Shape is (batch_size, n_top, 1)
            actions_indices = jnp.argsort(qf_next_values, axis=1, descending=True)[:, :n_top, :]
            # Shape (batch_size, n_top, action_dim)
            best_actions = jnp.take_along_axis(actions, actions_indices, axis=1)

            # Update centroid: barycenter of the best candidates
            return {
                "best_actions": best_actions.mean(axis=1),
                "best_actions_cov": best_actions.var(axis=1),
                "next_q_values": qf_next_values.max(axis=1),
            }

        update_carry = jax.lax.fori_loop(0, n_iterations, one_update, carry)
        # shape (batch_size, 1)
        return update_carry["next_q_values"]

    @staticmethod
    @jax.jit
    def update_qnetwork(
        qf_state: RLTrainState,
        observations: jax.Array,
        replay_actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        discounts: jax.Array,
        key: jax.Array,
        sampled_actions: jax.Array,
    ):
        # Uniform sampling
        next_actions = jax.random.uniform(
            key,
            shape=(observations.shape[0], sampled_actions.shape[0], replay_actions.shape[-1]),
            minval=-1.0,
            maxval=1.0,
        )
        # Gaussian dist
        # scale = 1.0
        # next_actions = scale * jax.random.normal(
        #     key,
        #     shape=(observations.shape[0], sampled_actions.shape[0], replay_actions.shape[-1]),
        # )
        # next_actions = jnp.clip(next_actions, -1.0, 1.0)

        repeated_next_obs = jnp.repeat(jnp.expand_dims(next_observations, axis=1), sampled_actions.shape[0], axis=1)

        # Compute the next Q-values using the target network
        qf_next_values = qf_state.apply_fn(qf_state.target_params, repeated_next_obs, next_actions)
        # Twin network: take the min between q-networks
        qf_next_values = jnp.min(qf_next_values, axis=0)

        # Follow greedy policy: use the one with the highest value
        next_q_values = qf_next_values.max(axis=1)

        # next_q_values = SampleDQN.find_max_target_q_cem(
        #     qf_state,
        #     observations,
        #     key,
        #     n_sampled_actions=sampled_actions.shape[0],
        #     action_dim=replay_actions.shape[-1],
        # )

        # shape is (batch_size, 1)
        target_q_values = rewards[:, None] + (1 - dones[:, None]) * discounts[:, None] * next_q_values

        def critic_loss(params):
            # Retrieve the q-values for the actions from the replay buffer
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf_state.apply_fn(params, observations, replay_actions)
            # Compute Huber loss (less sensitive to outliers)
            # return optax.huber_loss(current_q_values, target_q_values).mean(axis=1).sum(), current_q_values.mean()
            # Reduction: mean over batch, sum over critics
            return 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum(), current_q_values.mean()

        (qf_loss_value, qf_mean_value), grads = jax.value_and_grad(critic_loss, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads)

        return qf_state, (qf_loss_value, qf_mean_value)

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState):
        qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        return qf_state

    @staticmethod
    @jax.jit
    def _train(carry, indices):
        data = carry["data"]
        carry["key"], key = jax.random.split(carry["key"])

        qf_state, (qf_loss_value, qf_mean_value) = SampleDQN.update_qnetwork(
            carry["qf_state"],
            observations=data.observations[indices],
            replay_actions=data.actions[indices],
            next_observations=data.next_observations[indices],
            rewards=data.rewards[indices],
            dones=data.dones[indices],
            discounts=data.discounts[indices],
            key=key,
            sampled_actions=carry["sampled_actions"],
        )
        qf_state = SampleDQN.soft_update(carry["tau"], qf_state)

        carry["qf_state"] = qf_state
        carry["info"]["critic_loss"] += qf_loss_value
        carry["info"]["qf_mean_value"] += qf_mean_value

        return carry, None
