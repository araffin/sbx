from typing import Any, Optional

import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

# from https://github.com/ikostrikov/walk_in_the_park
# otherwise mode is not define for Squashed Gaussian
class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False):
        super().__init__(distribution=distribution, bijector=tfp.bijectors.Tanh(), validate_args=validate_args)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
