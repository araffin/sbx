from dataclasses import dataclass

import numpy as np


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
