from typing import Any, Optional

import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """
    From https://github.com/ikostrikov/walk_in_the_park
    otherwise mode is not defined for Squashed Gaussian
    """

    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class StateDependentNoiseDistribution(tfd.MultivariateNormalDiag):
    def __init__(self, loc=None, scale_diag=None, std=None, latent_sde=None):
        super().__init__(loc=loc, scale_diag=scale_diag)
        self.std = std
        self.latent_sde = latent_sde

    def sample(self, seed=None) -> jnp.ndarray:
        exploration_mat = tfd.MultivariateNormalDiag(
            loc=jnp.zeros_like(self.std),
            scale_diag=self.std,
        ).sample(seed=seed)
        return self.latent_sde @ exploration_mat
