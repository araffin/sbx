import warnings
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, is_vectorized_observation

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.dqn.policies import DQNPolicy


class DQN(OffPolicyAlgorithmJax):

    policy_aliases: Dict[str, Optional[nn.Module]] = {
        "MlpPolicy": DQNPolicy,
    }

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 1.0,
        gamma: float = 0.99,
        target_update_interval: int = 1000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        # max_grad_norm: float = 10,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
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
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(gym.spaces.Discrete),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

        if self.policy is None:
            self.policy = self.policy_class(  # pytype:disable=not-instantiable
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,  # pytype:disable=not-instantiable
            )

            self.key = self.policy.build(self.key, self.lr_schedule)
            self.qf = self.policy.qf

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
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
        # Convert to numpy
        data = ReplayBufferSamplesNp(
            data.observations.numpy(),
            # Convert to int64
            data.actions.long().numpy(),
            data.next_observations.numpy(),
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        self.policy.qf_state, self.key, qf_loss_value = self._train(
            self.qf,
            self.gamma,
            gradient_steps,
            data,
            self.policy.qf_state,
            self.key,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", qf_loss_value.item())

    @staticmethod
    @partial(jax.jit, static_argnames=["qf", "gamma"])
    def update_qnetwork(
        qf,
        gamma,
        qf_state: RLTrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key: jax.random.KeyArray,
    ):
        key, dropout_key_target, dropout_key_current = jax.random.split(key, 3)

        # Compute the next Q-values using the target network
        qf_next_values = qf.apply(
            qf_state.target_params,
            next_observations,
            rngs={"dropout": dropout_key_target},
        )
        # Follow greedy policy: use the one with the highest value
        next_q_values = qf_next_values.max(axis=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)

        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values

        def huber_loss(params, dropout_key):
            # Get current Q-values estimates
            current_q_values = qf.apply(params, observations, rngs={"dropout": dropout_key})
            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = jnp.take_along_axis(current_q_values, actions, axis=1)
            # Compute Huber loss (less sensitive to outliers)
            return optax.huber_loss(current_q_values, target_q_values).mean()

        qf_loss_value, grads = jax.value_and_grad(huber_loss, has_aux=False)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return qf_state, qf_loss_value, key

    @staticmethod
    @partial(jax.jit, static_argnames=["tau"])
    def soft_update(tau, qf_state: RLTrainState):
        qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        return qf_state

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            self.policy.qf_state = DQN.soft_update(self.tau, self.policy.qf_state)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
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
            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    @classmethod
    @partial(
        jax.jit,
        static_argnames=["cls", "qf", "gamma", "gradient_steps"],
    )
    def _train(
        cls,
        qf,
        gamma,
        gradient_steps,
        data: ReplayBufferSamplesNp,
        qf_state: RLTrainState,
        key,
    ):

        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]

            qf_state, qf_loss_value, key = DQN.update_qnetwork(
                qf,
                gamma,
                qf_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                key,
            )

        return qf_state, key, qf_loss_value
