import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from sbx.common.jax_layers import BatchRenorm


class _Net(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        x = BatchRenorm(use_running_average=not train, warmup_steps=5)(x)
        return nn.Dense(1)(x)


@pytest.mark.parametrize("steps", [0, 10])  # during and after warm-up
def test_batch_renorm_constant_feature_finite_grad(steps):
    # A feature that is constant over the batch (e.g. a dead ReLU unit or an action
    # saturated at the bound) has batch_var == 0, the gradient must stay finite
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (64, 4)).at[:, 0].set(1.0)

    net = _Net()
    variables = net.init(key, x, train=False)
    batch_stats = variables["batch_stats"]
    batch_stats["BatchRenorm_0"]["steps"] = jnp.asarray(steps)

    def loss(params, x):
        out, _ = net.apply({"params": params, "batch_stats": batch_stats}, x, train=True, mutable=["batch_stats"])
        return jnp.mean(out**2)

    grad_params, grad_x = jax.grad(loss, argnums=(0, 1))(variables["params"], x)
    assert jnp.all(jnp.isfinite(grad_x))
    for leaf in jax.tree_util.tree_leaves(grad_params):
        assert jnp.all(jnp.isfinite(leaf))
