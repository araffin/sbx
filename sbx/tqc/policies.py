from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
import tensorflow_probability

from sbx.common.distributions import TanhTransformedDistribution

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class Critic(nn.Module):
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_quantiles: int = 25
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.n_units)(x)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        if self.dropout_rate is not None and self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        if self.use_layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_quantiles)(x)
        return x


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        # dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist
