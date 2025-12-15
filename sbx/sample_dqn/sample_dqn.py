from functools import partial
from typing import Any, ClassVar

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
from sbx.sample_dqn.policies import NAME_TO_SAMPLING_STRATEGY, SampleDQNPolicy, SamplingStrategy


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
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0.99,
        n_sampled_actions: int = 25,
        action_noise: ActionNoise | None = None,
        # target_update_interval: int = 1000,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        train_sampling_strategy: SamplingStrategy | str = SamplingStrategy.UNIFORM,
        # max_grad_norm: float = 10,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
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

        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.00

        # TODO: handle error cases
        if isinstance(train_sampling_strategy, str):
            train_sampling_strategy = NAME_TO_SAMPLING_STRATEGY[train_sampling_strategy]
        self.train_sampling_strategy = train_sampling_strategy
        self.n_sampled_actions = n_sampled_actions
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            # Allow to have different n_sampled_actions before exploration and training
            if "n_sampled_actions" not in self.policy_kwargs:
                self.policy_kwargs["n_sampled_actions"] = self.n_sampled_actions

            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
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
            # "sampling_strategy": jnp.array([self.train_sampling_strategy.value]),
            "sampling_strategy": self.train_sampling_strategy.value,
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
        qf_loss_value = update_carry["info"]["critic_loss"] / gradient_steps
        qf_mean_value = update_carry["info"]["qf_mean_value"] / gradient_steps

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/qf_mean_value", qf_mean_value.item())

    @staticmethod
    @partial(jax.jit, static_argnames=["n_sampled_actions", "action_dim"])
    def find_max_target_q_cem(
        qf_state,
        next_observations,
        key,
        n_sampled_actions: int,
        action_dim: int,
        n_top: int = 6,
        n_iterations: int = 2,
        initial_variance: float = 1.0**2,
        extra_noise_std: float = 0.1,
    ):
        """
        Noisy Cross Entropy Method: http://dx.doi.org/10.1162/neco.2006.18.12.2936
        "Learning Tetris Using the Noisy Cross-Entropy Method"

        See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/62
        """
        best_actions = jnp.zeros((next_observations.shape[0], action_dim))
        best_actions_cov = jnp.ones_like(best_actions) * initial_variance
        extra_variance = jnp.ones_like(best_actions_cov) * extra_noise_std**2
        # Decay the extra noise in half the iterations
        extra_var_decay_time = n_iterations / 2.0

        carry = {
            "best_actions": best_actions,
            "best_actions_cov": best_actions_cov,
            "next_q_values": jnp.zeros((next_observations.shape[0], 1)),
            "key": key,
        }

        def one_update(i: int, carry: dict[str, Any]) -> dict[str, Any]:
            best_actions = carry["best_actions"]
            best_actions_cov = carry["best_actions_cov"]
            key = carry["key"]
            key, new_key = jax.random.split(key, 2)

            # Reduce extra variance over time
            extra_var_multiplier = jnp.max(jnp.array([(1.0 - i / extra_var_decay_time), 0]))
            # Sample using only the diagonal of the covariance matrix (+ extra noise)
            # TODO: try with full covariance?
            deltas = jax.random.normal(key, shape=(next_observations.shape[0], n_sampled_actions, action_dim))
            actions = jnp.expand_dims(best_actions, axis=1) + deltas * jnp.expand_dims(
                jnp.sqrt(best_actions_cov + extra_variance * extra_var_multiplier), axis=1
            )
            # actions = jnp.clip(actions, -1.0, 1.0)

            repeated_next_obs = jnp.repeat(jnp.expand_dims(next_observations, axis=1), n_sampled_actions, axis=1)

            # TD3 trick: note would not fully work here without re-eval the target
            # Select action according to target net and add clipped noise
            # target_policy_noise = 0.2
            # target_noise_clip = 0.5
            # noise = jax.random.normal(key, actions.shape) * target_policy_noise
            # noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
            # next_state_actions = jnp.clip(actions + noise, -1.0, 1.0)

            next_state_actions = jnp.clip(actions, -1.0, 1.0)

            # Shape is (n_critics, batch_size, n_repeated_actions, 1)
            qf_next_values = qf_state.apply_fn(qf_state.target_params, repeated_next_obs, next_state_actions)
            # Twin network: take the min between q-networks
            # qf_next_values = jnp.min(qf_next_values, axis=0)
            # More optimistic alternative
            qf_next_values = jnp.mean(qf_next_values, axis=0)

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
                "key": new_key,
            }

        update_carry = jax.lax.fori_loop(0, n_iterations, one_update, carry)
        # shape (batch_size, 1)
        return update_carry["next_q_values"]

    @staticmethod
    @partial(jax.jit, static_argnames=["n_sampled_actions", "action_dim"])
    def find_max_target_uniform(
        qf_state,
        next_observations,
        key,
        n_sampled_actions: int,
        action_dim: int,
        sampling_strategy: int = SamplingStrategy.UNIFORM.value,
    ):

        def uniform_sampling(key):
            return jax.random.uniform(
                key,
                shape=(next_observations.shape[0], n_sampled_actions, action_dim),
                minval=-1.0,
                maxval=1.0,
            )

        def gaussian_sampling(key):
            # Gaussian dist
            scale = 1.0
            next_actions = scale * jax.random.normal(
                key,
                shape=(next_observations.shape[0], n_sampled_actions, action_dim),
            )
            return jnp.clip(next_actions, -1.0, 1.0)

        next_actions = jax.lax.cond(
            sampling_strategy == SamplingStrategy.UNIFORM.value,
            # If True:
            uniform_sampling,
            # If False:
            gaussian_sampling,
            key,
        )

        repeated_next_obs = jnp.repeat(jnp.expand_dims(next_observations, axis=1), n_sampled_actions, axis=1)

        # Compute the next Q-values using the target network
        qf_next_values = qf_state.apply_fn(qf_state.target_params, repeated_next_obs, next_actions)
        # Twin network: take the min between q-networks
        # qf_next_values = jnp.min(qf_next_values, axis=0)
        # More optimistic alternative
        qf_next_values = jnp.mean(qf_next_values, axis=0)

        # Follow greedy policy: use the one with the highest value
        next_q_values = qf_next_values.max(axis=1)

        return next_q_values

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
        sampling_strategy: int = SamplingStrategy.UNIFORM.value,
    ):
        # Reduce number of sampled action compared to exploration
        # n_sampled_actions = sampled_actions.shape[0] // 2
        n_sampled_actions = sampled_actions.shape[0]
        action_dim = replay_actions.shape[-1]

        next_q_values = jax.lax.cond(
            sampling_strategy == SamplingStrategy.CEM.value,
            # If True:
            partial(SampleDQN.find_max_target_q_cem, n_sampled_actions=n_sampled_actions, action_dim=action_dim),
            # If False:
            partial(
                SampleDQN.find_max_target_uniform,
                n_sampled_actions=n_sampled_actions,
                action_dim=action_dim,
                sampling_strategy=sampling_strategy,
            ),
            qf_state,
            next_observations,
            key,
        )

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
            sampling_strategy=carry["sampling_strategy"],
        )
        qf_state = SampleDQN.soft_update(carry["tau"], qf_state)

        carry["qf_state"] = qf_state
        carry["info"]["critic_loss"] += qf_loss_value
        carry["info"]["qf_mean_value"] += qf_mean_value

        return carry, None

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state
