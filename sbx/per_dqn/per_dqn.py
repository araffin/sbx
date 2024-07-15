from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.prioritized_replay_buffer import PrioritizedReplayBuffer
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.dqn import DQN
from sbx.dqn.policies import CNNPolicy, DQNPolicy


class PERDQN(DQN):
    """
    DQN with Prioritized Experience Replay (PER).
    """

    policy_aliases: ClassVar[Dict[str, Type[DQNPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": DQNPolicy,
        "CnnPolicy": CNNPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    policy: DQNPolicy

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        optimize_memory_usage: bool = False,  # Note: unused but to match SB3 API
        # max_grad_norm: float = 10,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        # replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
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
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            optimize_memory_usage=optimize_memory_usage,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "PERDQN",
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

    def train(self, batch_size: int, gradient_steps: int) -> None:
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        # Convert to numpy
        data = ReplayBufferSamplesNp(
            data.observations.numpy(),
            # Convert to int64
            data.actions.long().numpy(),
            data.next_observations.numpy(),
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
            data.weights.numpy().flatten(),
            data.leaf_nodes_indices,
        )
        # Pre compute the slice indices
        # otherwise jax will complain
        indices = jnp.arange(len(data.dones)).reshape(gradient_steps, batch_size)

        update_carry = {
            "qf_state": self.policy.qf_state,
            "gamma": self.gamma,
            "data": data,
            "indices": indices,
            "info": {
                "critic_loss": jnp.array([0.0]),
                "qf_mean_value": jnp.array([0.0]),
                "td_error": jnp.zeros_like(data.rewards),
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
        qf_loss_value = update_carry["info"]["critic_loss"]
        qf_mean_value = update_carry["info"]["qf_mean_value"] / gradient_steps
        td_error = update_carry["info"]["td_error"]

        # Update priorities, they will be proportional to the td error
        # Note: compared to the original implementation, we update
        # the priorities after all the gradient steps
        self.replay_buffer.update_priorities(data.leaf_nodes_indices, td_error, self._current_progress_remaining)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/qf_mean_value", qf_mean_value.item())

    @staticmethod
    @jax.jit
    def update_qnetwork(
        gamma: float,
        qf_state: RLTrainState,
        observations: np.ndarray,
        replay_actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        sampling_weights: np.ndarray,
    ):
        # Compute the next Q-values using the target network
        qf_next_values = qf_state.apply_fn(qf_state.target_params, next_observations)

        # Follow greedy policy: use the one with the highest value
        next_q_values = qf_next_values.max(axis=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)

        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values

        # Special case when using PrioritizedReplayBuffer (PER)
        def weighted_huber_loss(params):
            # Get current Q-values estimates
            current_q_values = qf_state.apply_fn(params, observations)
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = jnp.take_along_axis(current_q_values, replay_actions, axis=1)
            # TD error in absolute value, to update priorities
            td_error = jnp.abs(current_q_values - target_q_values)
            # Weighted Huber loss using importance sampling weights
            loss = (sampling_weights * optax.huber_loss(current_q_values, target_q_values)).mean()
            return loss, (current_q_values.mean(), td_error.flatten())

        (qf_loss_value, (qf_mean_value, td_error)), grads = jax.value_and_grad(weighted_huber_loss, has_aux=True)(
            qf_state.params
        )
        qf_state = qf_state.apply_gradients(grads=grads)

        return qf_state, (qf_loss_value, qf_mean_value, td_error)

    @staticmethod
    @jax.jit
    def _train(carry, indices):
        data = carry["data"]

        qf_state, (qf_loss_value, qf_mean_value, td_error) = PERDQN.update_qnetwork(
            carry["gamma"],
            carry["qf_state"],
            observations=data.observations[indices],
            replay_actions=data.actions[indices],
            next_observations=data.next_observations[indices],
            rewards=data.rewards[indices],
            dones=data.dones[indices],
            sampling_weights=data.weights[indices],
        )

        carry["qf_state"] = qf_state
        carry["info"]["critic_loss"] += qf_loss_value
        carry["info"]["qf_mean_value"] += qf_mean_value
        carry["info"]["td_error"] = carry["info"]["td_error"].at[indices].set(td_error)

        return carry, None
