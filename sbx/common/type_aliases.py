from typing import NamedTuple, Union

import flax
import numpy as np
from flax.training.train_state import TrainState


class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]


class BatchNormTrainState(TrainState):  # type: ignore[misc]
    batch_stats: flax.core.FrozenDict  # type: ignore[misc]


class ReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
