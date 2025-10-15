from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax.training.train_state import TrainState


@dataclass
class KLAdaptiveLR:
    """Adaptive lr schedule, see https://arxiv.org/abs/1707.02286"""

    # If set will trigger adaptive lr
    target_kl: float
    current_adaptive_lr: float
    # Values taken from https://github.com/leggedrobotics/rsl_rl
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2
    kl_margin: float = 2.0
    # Divide or multiply the lr by this factor
    adaptive_lr_factor: float = 1.5

    def update(self, kl_div: float) -> None:
        if kl_div > self.target_kl * self.kl_margin:
            self.current_adaptive_lr /= self.adaptive_lr_factor
        elif kl_div < self.target_kl / self.kl_margin:
            self.current_adaptive_lr *= self.adaptive_lr_factor

        self.current_adaptive_lr = np.clip(self.current_adaptive_lr, self.min_learning_rate, self.max_learning_rate)


def mask_from_prefix(params: FrozenDict, prefix: str = "NatureCNN_") -> dict:
    """
    Build a pytree mask (same structure as `params`) where a leaf is True
    if the top-level module name starts with `prefix`.
    """

    def _traverse(tree: FrozenDict, path: tuple[str, ...] = ()) -> Union[dict, bool]:
        if isinstance(tree, dict):
            return {key: _traverse(value, (*path, key)) for key, value in tree.items()}
        # leaf
        return path[1].startswith(prefix) if len(path) > 1 else False

    return _traverse(params)  # type: ignore[return-value]


def align_params(params1: FrozenDict, params2: FrozenDict) -> dict:
    """
    Return a dict with the *exact* structure of `params2`. For every leaf in `params2`,
    use the corresponding leaf from `params1` if it exists; otherwise use `params2`'s leaf.
    This guarantees the two dict have identical structure for tree_map.
    """
    if isinstance(params2, dict):
        out = {}
        for key, params2_sub in params2.items():
            params1_sub = params1[key] if (isinstance(params1, dict) and key in params1) else None
            out[key] = align_params(params1_sub, params2_sub)  # type: ignore[arg-type]
        return out
    # leaf-case: if params1 value exists (not None and same shape) use it, else use params2 leaf
    return params1 if (params1 is not None and params1.shape == params2.shape) else params2  # type: ignore[attr-defined, return-value]


@jax.jit
def masked_copy(params1: FrozenDict, params2: FrozenDict, mask: dict) -> FrozenDict:
    """
    Leafwise selection: wherever mask is True we take params1 value,
    otherwise params2 value.
    """
    return jax.tree_util.tree_map(
        lambda val1, val2, mask_value: jnp.where(mask_value, val1, val2),
        params1,
        params2,
        mask,
    )


@jax.jit
def copy_naturecnn_params(state1: TrainState, state2: TrainState) -> TrainState:
    """
    Copy all top-level modules whose names start with "NatureCNN_" from
    state1.params into state2.params.
    It is useful when sharing features extractor parameters between actor and critic.
    """
    # Ensure same structure
    aligned_params = align_params(state1.params, state2.params)
    mask = mask_from_prefix(state2.params, prefix="NatureCNN_")
    new_params = masked_copy(aligned_params, state2.params, mask)

    return state2.replace(params=new_params)
