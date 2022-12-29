from typing import NamedTuple

import flax
import numpy as np
from flax.training.train_state import TrainState


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict


class ReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray
