from functools import partial
from typing import Callable, Generator, Optional, Tuple, Union, NamedTuple

import numpy as np
import torch as th
import jax.numpy as jnp
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

# class LSTMStates(NamedTuple):
#     pi: Tuple[th.Tensor, ...]
#     vf: Tuple[th.Tensor, ...]

# TODO : see if I add jax info
class LSTMStates(NamedTuple):
    pi: Tuple
    vf: Tuple

# Replaced th.Tensor with jnp.ndarray
class RecurrentRolloutBufferSamples(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    old_values: jnp.ndarray
    old_log_prob: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    lstm_states: LSTMStates

class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        # renamed this because I found hidden_state_shape confusing
        lstm_state_buffer_shape: Tuple[int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):  
        # TODO : see if I rename this in all the code
        self.hidden_state_shape = lstm_state_buffer_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self):
        super().reset()
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)

    # def add(self, *args, lstm_states: LSTMStates, **kwargs) -> None:
    #     """
    #     :param hidden_states: LSTM cell and hidden state
    #     """
    #     # TODO : at the moment doesn't work because I didn't create a named tuple for lstm states
    #     self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
    #     self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
    #     self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
    #     self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

    def add(self, *args, lstm_states, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        """
        # TODO : at the moment doesn't work because I didn't create a named tuple for lstm states
        print(lstm_states[0][0])
        print(np.array(lstm_states[0][0]))
        print(np.array(lstm_states[0][0]).shape)
        print(self.hidden_states_pi[self.pos].shape)
        self.hidden_states_pi[self.pos] = np.array(lstm_states[0][0])
        self.cell_states_pi[self.pos] = np.array(lstm_states[0][1])
        self.hidden_states_vf[self.pos] = np.array(lstm_states[1][0])
        self.cell_states_vf[self.pos] = np.array(lstm_states[1][1])

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that doesn't allow any mini batch size (must be a product of n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds)
            start_idx += batch_size


    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        
        lstm_states_pi = (
            self.hidden_states_pi[batch_inds],
            self.cell_states_pi[batch_inds]
        )

        lstm_states_vf = (
            self.hidden_states_vf[batch_inds],
            self.cell_states_vf[batch_inds]
        )

        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            LSTMStates(pi=lstm_states_pi, vf=lstm_states_vf)
        )
        return RecurrentRolloutBufferSamples(*tuple(map(self.to_torch, data)))
