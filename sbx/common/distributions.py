from typing import Any

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


class TanhTransformedDistribution(tfd.TransformedDistribution):  # type: ignore[name-defined]
    """
    From https://github.com/ikostrikov/walk_in_the_park
    otherwise mode is not defined for Squashed Gaussian
    """

    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):  # type: ignore[name-defined]
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Any | None, num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
