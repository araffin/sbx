"""
Segment tree implementation taken from Stable Baselines 2:
https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/segment_tree.py

Notable differences:
- This implementation uses numpy arrays to store the values (faster initialization)
- We don't use a special function to have unique indices (no significant performance difference found)
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


class SegmentTree:
    def __init__(self, capacity: int, reduce_op: Callable, neutral_element: float) -> None:
        """
        Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array that supports Index arrays, but with two
        important differences:

        a) setting item's value is slightly slower.
            It is O(log capacity) instead of O(1).
        b) user has access to an efficient ( O(log segment size) )
            `reduce` operation which reduces `operation` over
            a contiguous subsequence of items in the array.

        :param capacity: Total size of the array - must be a power of two.
        :param reduce_op: Operation for combining elements (eg. sum, max) must form a
            mathematical group together with the set of possible values for array elements (i.e. be associative)
        :param neutral_element: Neutral element for the operation above. eg. float('-inf') for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, f"Capacity must be positive and a power of 2, not {capacity}"
        self._capacity = capacity
        # First index is the root, leaf nodes are in [capacity, 2 * capacity - 1].
        # For each parent node i, left child has index [2 * i], right child [2 * i + 1]
        self._values = np.full(2 * capacity, neutral_element)
        self._reduce_op = reduce_op
        self.neutral_element = neutral_element

    def _reduce_helper(self, start: int, end: int, node: int, node_start: int, node_end: int) -> float:
        """
        Query the value of the segment tree for the given range

        :param start: start of the range
        :param end: end of the range
        :param node: current node in the segment tree
        :param node_start: start of the range represented by the current node
        :param node_end: end of the range represented by the current node
        :return: result of reducing ``self.reduce_op`` over the specified range of array elements.
        """
        if start == node_start and end == node_end:
            return self._values[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._reduce_op(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Returns result of applying ``self.reduce_op``
        to a contiguous subsequence of the array.

        .. code-block:: python

            self.reduce_op(arr[start], operation(arr[start+1], operation(... arr[end])))

        :param start: beginning of the subsequence
        :param end: end of the subsequences
        :return: result of reducing ``self.reduce_op`` over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx: np.ndarray, val: np.ndarray) -> None:
        """
        Set the value at index `idx` to `val`

        :param idx: index of the value to be updated
        :param val: new value
        """
        # assert np.all(0 <= idx < self._capacity), f"Trying to set item outside capacity: {idx}"
        # Indices of the leafs
        indices = idx + self._capacity
        # Update the leaf nodes and then the related nodes
        self._values[indices] = val
        if isinstance(indices, int):
            indices = np.array([indices])
        # Go up one level in the tree and remove duplicate indices
        indices = np.unique(indices // 2)
        while len(indices) > 1 or indices[0] > 0:
            # As long as there are non-zero indices, update the corresponding values
            self._values[indices] = self._reduce_op(self._values[2 * indices], self._values[2 * indices + 1])
            # Go up one level in the tree and remove duplicate indices
            indices = np.unique(indices // 2)

    def __getitem__(self, idx: np.ndarray) -> np.ndarray:
        """
        Get the value(s) at index `idx`
        """
        assert np.max(idx) < self._capacity, f"Index must be less than capacity, got {np.max(idx)} >= {self._capacity}"
        assert 0 <= np.min(idx)
        return self._values[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    """
    A Segment Tree data structure where each node contains the sum of the
    values in its leaf nodes. Can be used as a Sum Tree for priorities.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, reduce_op=np.add, neutral_element=0.0)

    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Returns arr[start] + ... + arr[end]

        :param start: start position of the reduction (must be >= 0)
        :param end: end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: reduction of SumSegmentTree
        """
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: np.ndarray) -> np.ndarray:
        """
        Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum for each entry in prefixsum

        if array values are probabilities, this function
        allows to sample indices according to the discrete
        probability efficiently.

        :param prefixsum: float upper bounds on the sum of array prefix
        :return: highest indices satisfying the prefixsum constraint
        """
        if isinstance(prefixsum, float):
            prefixsum = np.array([prefixsum])
        assert 0 <= np.min(prefixsum)
        # assert np.max(prefixsum) <= self.sum() + 1e-5

        indices = np.ones(len(prefixsum), dtype=int)
        should_continue = np.ones(len(prefixsum), dtype=bool)

        while np.any(should_continue):  # while not all nodes are leafs
            indices[should_continue] = 2 * indices[should_continue]
            prefixsum_new = np.where(
                self._values[indices] <= prefixsum,
                prefixsum - self._values[indices],
                prefixsum,
            )
            # Prepare update of prefixsum for all right children
            indices = np.where(
                np.logical_or(self._values[indices] > prefixsum, np.logical_not(should_continue)),
                indices,
                indices + 1,
            )
            # Select child node for non-leaf nodes
            prefixsum = prefixsum_new
            # Update prefixsum
            should_continue = indices < self._capacity
        # Collect leafs
        return indices - self._capacity


class MinSegmentTree(SegmentTree):
    """
    A Segment Tree data structure where each node contains the minimum of the
    values in its leaf nodes. Can be used as a Min Tree for priorities.
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity=capacity, reduce_op=np.minimum, neutral_element=float("inf"))

    def min(self, start=0, end=None):
        """
        Returns min(arr[start], ...,  arr[end])

        :param start: start position of the reduction (must be >= 0)
        :param end: end position of the reduction (must be < len(arr), can be None for len(arr) - 1)
        :return: reduction of MinSegmentTree
        """
        return super().reduce(start, end)


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
        optimize_memory_usage: bool = False,
        min_priority: float = 1e-6,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)

        # TODO: check if we can support optimize_memory_usage
        assert optimize_memory_usage is False, "PrioritizedReplayBuffer doesn't support optimize_memory_usage=True"

        # TODO: add support for multi env
        # assert n_envs == 1, "PrioritizedReplayBuffer doesn't support n_envs > 1"

        # Find the next power of 2 for the buffer size
        power_of_two = int(np.ceil(np.log2(buffer_size)))
        tree_capacity = 2**power_of_two

        self._min_priority = min_priority
        self._max_priority = 1.0

        self._alpha = alpha

        self._sum_tree = SumSegmentTree(tree_capacity)
        self._min_tree = MinSegmentTree(tree_capacity)
        # Flatten the indices from the buffer to store them in the sum tree
        # Replay buffer: (idx, env_idx)
        # Sum tree: idx * self.n_envs + env_idx
        self.env_offsets = np.arange(self.n_envs)

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
        # Store transition index with maximum priority in sum tree
        self._sum_tree[self.pos * self.n_envs + self.env_offsets] = self._max_priority**self._alpha
        self._min_tree[self.pos * self.n_envs + self.env_offsets] = self._max_priority**self._alpha

        # Store transition in the buffer
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, beta: float, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the prioritized replay buffer.

        :param batch_size: Number of element to sample
        :param env:associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: a batch of sampled experiences from the buffer.
        """
        # assert self.buffer_size >= batch_size, "The buffer contains less samples than the batch size requires."

        # priorities = np.zeros((batch_size, 1))
        # sample_indices = np.zeros(batch_size, dtype=np.uint32)

        # TODO: check how things are sampled in the original implementation

        leaf_nodes_indices = self._sample_proportional(batch_size)
        # Convert the leaf nodes indices to buffer indices
        # Replay buffer: (idx, env_idx)
        # Sum tree: idx * self.n_envs + env_idx
        buffer_indices = leaf_nodes_indices // self.n_envs
        env_indices = leaf_nodes_indices % self.n_envs

        # probability of sampling transition i as P(i) = p_i^alpha / \sum_{k} p_k^alpha
        # where p_i > 0 is the priority of transition i.
        # probs = priorities / self.tree.total_sum
        probabilities = self._sum_tree[leaf_nodes_indices] / self._sum_tree.sum()

        # Importance sampling weights.
        # All weights w_i were scaled so that max_i w_i = 1.
        # weights = (self.size() * probs + 1e-7) ** -self.beta
        # min_probability = self._min_tree.min() / self._sum_tree.sum()
        # max_weight = (min_probability * self.size()) ** (-self.beta)
        # weights = (probabilities * self.size()) ** (-self.beta) / max_weight
        weights = (probabilities * self.size()) ** (-beta)
        weights = weights / weights.max()

        # env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size,))
        # env_indices = np.zeros(batch_size, dtype=np.uint32)
        next_obs = self._normalize_obs(self.next_observations[buffer_indices, env_indices, :], env)

        batch = (
            self._normalize_obs(self.observations[buffer_indices, env_indices, :], env),
            self.actions[buffer_indices, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[buffer_indices, env_indices] * (1 - self.timeouts[buffer_indices, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[buffer_indices, env_indices].reshape(-1, 1), env),
            weights,
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, batch)), leaf_nodes_indices)  # type: ignore[arg-type,call-arg]

    def _sample_proportional(self, batch_size: int) -> np.ndarray:
        """
        Sample a batch of leaf nodes indices using the proportional prioritization strategy.
        In other words, the probability of sampling a transition is proportional to its priority.

        :param batch_size: Number of element to sample
        :return: Indices of the sampled leaf nodes
        """
        # TODO: double check if this is correct
        total = self._sum_tree.sum(0, self.size() - 1)
        priorities_sum = np.random.random(size=batch_size) * total
        return self._sum_tree.find_prefixsum_idx(priorities_sum)

    # def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
    def update_priorities(self, leaf_nodes_indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities of sampled transitions.

        :param leaf_nodes_indices: Indices of the sampled transitions.
        :param priorities: New priorities, td error in the case of
            proportional prioritized replay buffer.
        """
        # TODO: double check that all samples are updated
        # priorities = np.abs(td_errors) + self.min_priority
        priorities += self._min_priority
        # assert len(indices) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(leaf_nodes_indices) >= 0
        assert np.max(leaf_nodes_indices) < self.buffer_size
        # TODO: check if we need to add the min_priority here
        # priorities = (np.abs(td_errors) + self.min_priority) ** self.alpha
        self._sum_tree[leaf_nodes_indices] = priorities**self._alpha
        self._min_tree[leaf_nodes_indices] = priorities**self._alpha
        # Update max priority for new samples
        self._max_priority = max(self._max_priority, np.max(priorities))
