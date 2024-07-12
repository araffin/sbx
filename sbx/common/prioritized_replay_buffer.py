# from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class SumTree:
    """
    SumTree: a binary tree data structure where the parent's value is the sum of its children.
    """

    def __init__(self, buffer_size: int, rng_key: jax.Array):
        self.buffer_size = buffer_size
        self.tree = jnp.zeros(2 * buffer_size - 1)
        self.data = jnp.zeros(buffer_size, dtype=jnp.float32)
        self.size = 0
        self.key = rng_key

    @staticmethod
    # TODO: try forcing on cpu
    # partial(jax.jit, backend="cpu")
    @jax.jit
    def _add(
        tree: jnp.ndarray,
        data: jnp.ndarray,
        size: int,
        capacity: int,
        priority: float,
        new_data: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
        index = size + capacity - 1
        data = data.at[size].set(new_data)
        tree = SumTree._update(tree, index, priority)
        size += 1
        return tree, data, size

    def add(self, priority: float, new_data: int) -> None:
        """
        Add a new transition with priority value,
        it adds a new leaf node and update cumulative sum.

        :param priority: Priority value.
        :param new_data: Data for the new leaf node, storing transition index
            in the case of the prioritized replay buffer.
        """
        self.tree, self.data, self.size = self._add(self.tree, self.data, self.size, self.buffer_size, priority, new_data)

    @staticmethod
    @jax.jit
    def _update(tree: jnp.ndarray, index: int, priority: float) -> jnp.ndarray:
        change = priority - tree[index]
        tree = tree.at[index].set(priority)
        tree = SumTree._propagate(tree, index, change)
        return tree

    def update(self, leaf_node_idx: int, priority: float) -> None:
        self.tree = self._update(self.tree, leaf_node_idx, priority)

    @staticmethod
    @jax.jit
    def _propagate(tree: jnp.ndarray, index: int, change: float) -> jnp.ndarray:
        def cond_fun(val) -> bool:
            idx, _, _ = val
            return idx > 0

        def body_fun(val) -> Tuple[int, float, jnp.ndarray]:
            idx, change, tree = val
            parent = (idx - 1) // 2
            tree = tree.at[parent].add(change)
            return parent, change, tree

        _, _, tree = jax.lax.while_loop(cond_fun, body_fun, (index, change, tree))
        return tree

    @property
    def total_sum(self) -> float:
        return self.tree[0].item()

    @staticmethod
    @jax.jit
    def _get(
        tree: jnp.ndarray,
        data: jnp.ndarray,
        capacity: int,
        priority_sum: float,
    ) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
        index = SumTree._retrieve(tree, priority_sum)
        data_index = index - capacity + 1
        return index, tree[index], data[data_index]

    def get(self, cumulative_sum: float) -> Tuple[int, float, int]:
        """
        Get a leaf node index, its priority value and transition index by cumulative_sum value.

        :param cumulative_sum: Cumulative sum value.
        :return: Leaf node index, its priority value and transition index.
        """
        leaf_tree_index, priority, transition_index = self._get(self.tree, self.data, self.buffer_size, cumulative_sum)
        return leaf_tree_index, priority.item(), transition_index.item()

    @staticmethod
    @jax.jit
    def _retrieve(tree: jnp.ndarray, priority_sum: float) -> int:
        def cond_fun(args) -> bool:
            idx, _ = args
            left = 2 * idx + 1
            return left < len(tree)

        def body_fun(args) -> Tuple[int, float]:
            idx, priority_sum = args
            left = 2 * idx + 1
            right = left + 1

            def left_branch(_) -> Tuple[int, float]:
                return left, priority_sum

            def right_branch(_) -> Tuple[int, float]:
                return right, priority_sum - tree[left]

            idx, priority_sum = jax.lax.cond(priority_sum <= tree[left], left_branch, right_branch, None)
            return idx, priority_sum

        index, _ = jax.lax.while_loop(cond_fun, body_fun, (0, priority_sum))
        return index

    @staticmethod
    @jax.jit
    def _batch_update(
        tree: jnp.ndarray,
        leaf_nodes_indices: jnp.ndarray,
        priorities: jnp.ndarray,
    ) -> jnp.ndarray:
        for leaf_node_idx, priority in zip(leaf_nodes_indices, priorities):
            tree = SumTree._update(tree, leaf_node_idx, priority)
        return tree

    def batch_update(self, leaf_nodes_indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Batch update transition priorities.

        :param leaf_nodes_indices: Indices for the leaf nodes to update
            (correponding to the transitions)
        :param priorities: New priorities, td error in the case of
            proportional prioritized replay buffer.
        """
        self.tree = self._batch_update(self.tree, leaf_nodes_indices, priorities)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer (proportional priorities version).
    Paper: https://arxiv.org/abs/1511.05952
    This code is inspired by: https://github.com/Howuhh/prioritized_experience_replay

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param alpha: How much prioritization is used (0 - no prioritization aka uniform case, 1 - full prioritization)
    :param beta: To what degree to use importance weights (0 - no corrections, 1 - full correction)
    :param final_beta: Value of beta at the end of training.
        Linear annealing is used to interpolate between initial value of beta and final beta.
    :param min_priority: Minimum priority, prevents zero probabilities, so that all samples
        always have a non-zero probability to be sampled.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        alpha: float = 0.5,
        beta: float = 0.4,
        final_beta: float = 1.0,
        optimize_memory_usage: bool = False,
        min_priority: float = 1e-6,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)

        assert optimize_memory_usage is False, "PrioritizedReplayBuffer doesn't support optimize_memory_usage=True"

        self.min_priority = min_priority
        self.alpha = alpha
        self.max_priority = self.min_priority  # priority for new samples, init as eps
        # Track the training progress remaining (from 1 to 0)
        # this is used to update beta
        self._current_progress_remaining = 1.0
        self.inital_beta = beta
        self.final_beta = final_beta
        self.beta_schedule = get_linear_fn(
            self.inital_beta,
            self.final_beta,
            end_fraction=1.0,
        )
        # SumTree: data structure to store priorities
        self.tree = SumTree(buffer_size=buffer_size, rng_key=jax.random.PRNGKey(0))

    @property
    def beta(self) -> float:
        # Linear schedule
        return self.beta_schedule(self._current_progress_remaining)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new transition to the buffer.

        :param obs: Starting observation of the transition to be stored.
        :param next_obs: Destination observation of the transition to be stored.
        :param action: Action performed in the transition to be stored.
        :param reward: Reward received in the transition to be stored.
        :param done: Whether the episode was finished after the transition to be stored.
        :param infos: Eventual information given by the environment.
        """
        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.pos)

        # store transition in the buffer
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the prioritized replay buffer.

        :param batch_size: Number of element to sample
        :param env:associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: a batch of sampled experiences from the buffer.
        """
        assert self.buffer_size >= batch_size, "The buffer contains less samples than the batch size requires."

        leaf_nodes_indices = np.zeros(batch_size, dtype=np.uint32)
        priorities = np.zeros((batch_size, 1))
        sample_indices = np.zeros(batch_size, dtype=np.uint32)

        # To sample a minibatch of size k, the range [0, total_sum] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree.
        segment_size = self.tree.total_sum / batch_size
        for batch_idx in range(batch_size):
            # extremes of the current segment
            start, end = segment_size * batch_idx, segment_size * (batch_idx + 1)

            # uniformely sample a value from the current segment
            cumulative_sum = np.random.uniform(start, end)

            # leaf_node_idx is a index of a sample in the tree, needed further to update priorities
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            leaf_node_idx, priority, sample_idx = self.tree.get(cumulative_sum)

            leaf_nodes_indices[batch_idx] = leaf_node_idx
            priorities[batch_idx] = priority
            sample_indices[batch_idx] = sample_idx

        # sample_indices, priorities, leaf_nodes_indices = self.tree.stratified_sampling(batch_size)

        # probability of sampling transition i as P(i) = p_i^alpha / \sum_{k} p_k^alpha
        # where p_i > 0 is the priority of transition i.
        probs = priorities / self.tree.total_sum

        # Importance sampling weights.
        # All weights w_i were scaled so that max_i w_i = 1.
        weights = (self.size() * probs + 1e-7) ** -self.beta
        weights = weights / weights.max()

        # TODO: add proper support for multi env
        # env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size,))
        env_indices = np.zeros(batch_size, dtype=np.uint32)

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(sample_indices + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[sample_indices, env_indices, :], env)

        batch = (
            self._normalize_obs(self.observations[sample_indices, env_indices, :], env),
            self.actions[sample_indices, env_indices, :],
            next_obs,
            self.dones[sample_indices],
            self.rewards[sample_indices],
            weights,
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, batch)), leaf_nodes_indices)  # type: ignore[arg-type,call-arg]

    def update_priorities(self, leaf_nodes_indices: np.ndarray, td_errors: np.ndarray, progress_remaining: float) -> None:
        """
        Update transition priorities.

        :param leaf_nodes_indices: Indices for the leaf nodes to update
            (correponding to the transitions)
        :param td_errors: New priorities, td error in the case of
            proportional prioritized replay buffer.
        :param progress_remaining: Current progress remaining (starts from 1 and ends to 0)
            to linearly anneal beta from its start value to 1.0 at the end of training
        """
        # Update beta schedule
        self._current_progress_remaining = progress_remaining

        # Batch update
        priorities = (np.abs(td_errors) + self.min_priority) ** self.alpha
        self.tree.batch_update(leaf_nodes_indices, priorities)
        # Update max priority for new samples
        self.max_priority = max(self.max_priority, priorities.max())
