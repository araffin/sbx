from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import optax


def update_learning_rate(opt_state: optax.OptState, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: Optax optimizer state
    :param learning_rate: New learning rate value
    """
    # Note: the optimizer must have been defined with inject_hyperparams
    opt_state.hyperparams["learning_rate"] = learning_rate


def kl_div_gaussian(
    old_std: jnp.ndarray,
    old_mean: jnp.ndarray,
    new_std: jnp.ndarray,
    new_mean: jnp.ndarray,
    eps: float = 1e-5,
) -> float:
    # See https://stats.stackexchange.com/questions/7440/
    # We have independent Gaussian for each action dim
    return (
        jnp.sum(
            jnp.log(new_std / old_std + eps) + ((old_std) ** 2 + (old_mean - new_mean) ** 2) / (2.0 * (new_std) ** 2) - 0.5,
            axis=-1,
        )
        .mean()
        .item()
    )


@dataclass
class KlAdaptiveLR:
    # If set will trigger adaptive lr
    target_kl: float
    current_adaptive_lr: float
    # Values taken from https://github.com/leggedrobotics/rsl_rl
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2
    kl_margin: float = 2.0
    # Divide or multiple the lr by this factor
    adaptive_lr_factor: float = 1.5

    def update(self, kl_div: float) -> None:
        if kl_div > self.target_kl * self.kl_margin:
            self.current_adaptive_lr /= self.adaptive_lr_factor
        elif kl_div < self.target_kl / self.kl_margin:
            self.current_adaptive_lr *= self.adaptive_lr_factor

        self.current_adaptive_lr = np.clip(self.current_adaptive_lr, self.min_learning_rate, self.max_learning_rate)
