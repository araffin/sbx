from dataclasses import dataclass

import jax
import jax.numpy as jnp


# JIT compatible version
@jax.tree_util.register_dataclass
@dataclass
class KLAdaptiveLR:
    """Adaptive lr schedule, see https://arxiv.org/abs/1707.02286"""

    # If set will trigger adaptive lr
    target_kl: float
    initial_lr: float
    # Values taken from https://github.com/leggedrobotics/rsl_rl
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-2
    kl_margin: float = 2.0
    # Divide or multiply the lr by this factor
    adaptive_lr_factor: float = 1.5
    current_adaptive_lr: float = 0.0

    def __post_init__(self) -> None:
        self.current_adaptive_lr = self.initial_lr

    def update(self, kl_div: float) -> float:
        self.current_adaptive_lr = jax.lax.select(  # type: ignore[assignment]
            kl_div > self.target_kl * self.kl_margin,
            # If True:
            self.current_adaptive_lr / self.adaptive_lr_factor,
            # If False:
            self.current_adaptive_lr,
        )

        self.current_adaptive_lr = jax.lax.select(  # type: ignore[assignment]
            kl_div < self.target_kl / self.kl_margin,
            # If True:
            self.current_adaptive_lr * self.adaptive_lr_factor,
            # If False:
            self.current_adaptive_lr,
        )

        self.current_adaptive_lr = jnp.clip(self.current_adaptive_lr, self.min_learning_rate, self.max_learning_rate)  # type: ignore[assignment]
        return self.current_adaptive_lr


@jax.jit
def adaptive_kl_lr(
    current_lr: float,
    target_kl: float,
    kl_div: float,
    min_learning_rate: float = 1e-5,
    max_learning_rate: float = 1e-2,
    # TODO: use percentage of target_kl instead of constant factor?
    kl_margin: float = 2.0,
    # Divide or multiply the lr by this factor
    adaptive_lr_factor: float = 1.5,
) -> jax.Array:

    # See https://stackoverflow.com/questions/75071836/
    branches = [lambda: current_lr / adaptive_lr_factor, lambda: current_lr * adaptive_lr_factor, lambda: current_lr]
    conditions = jnp.array([kl_div > target_kl * kl_margin, kl_div < (target_kl / kl_margin), True])
    new_lr = jax.lax.switch(jnp.argmax(conditions), branches)
    return jnp.clip(new_lr, min_learning_rate, max_learning_rate)
