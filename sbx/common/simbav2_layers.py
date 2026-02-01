from collections.abc import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from sbx.common.distributions import TanhTransformedDistribution
from sbx.common.policies import Flatten, VectorCritic, tfd

EPS = 1e-8


def l2normalize(x: jnp.ndarray, axis: int) -> jnp.ndarray:
    l2_norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    return x / jnp.maximum(l2_norm, EPS)


def l2normalize_layer(tree):
    """
    Apply l2-normalization to the all leaf nodes
    """
    if len(tree["kernel"].shape) == 2:
        axis = 0
    elif len(tree["kernel"].shape) == 3:
        axis = 1
    else:
        raise ValueError(f"Not supported tree: {tree}")
    return jax.tree.map(f=lambda x: l2normalize(x, axis=axis), tree=tree)


class Scaler(nn.Module):
    dim: int
    init_scale: float = 1.0
    scale: float = 1.0

    def setup(self):
        self.scaler = self.param("scaler", nn.initializers.constant(1.0 * self.scale), self.dim)
        self.forward_scaler = self.init_scale / self.scale

    def __call__(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    hidden_dim: int
    use_bias: bool = False  # important!

    def setup(self):
        self.linear = nn.Dense(
            name="hyper_dense",
            features=self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(scale=1.0, column_axis=0),
            use_bias=self.use_bias,
        )

    def __call__(self, x):
        return self.linear(x)


class HyperMLP(nn.Module):
    hidden_dim: int
    out_dim: int
    scaler_init: float
    scaler_scale: float
    eps: float = 1e-8
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def setup(self):
        self.layer1 = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)
        self.layer2 = HyperDense(self.out_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.scaler(self.layer1(x))
        # `eps` is required to prevent zero vector.
        x = self.activation_fn(x) + self.eps
        x = self.layer2(x)
        return l2normalize(x, axis=-1)


class HyperEmbedder(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    constant_shift: float

    def setup(self):
        self.dense = HyperDense(self.hidden_dim)
        self.scaler = Scaler(self.hidden_dim, self.scaler_init, self.scaler_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        new_axis = jnp.ones((*x.shape[:-1], 1)) * self.constant_shift
        x = jnp.concatenate([x, new_axis], axis=-1)
        x = l2normalize(x, axis=-1)
        x = self.scaler(self.dense(x))
        return l2normalize(x, axis=-1)


class HyperLERPBlock(nn.Module):
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    expansion: int = 4

    def setup(self):
        self.hyper_mlp = HyperMLP(
            hidden_dim=self.hidden_dim * self.expansion,
            out_dim=self.hidden_dim,
            scaler_init=self.scaler_init / np.sqrt(self.expansion),
            scaler_scale=self.scaler_scale / np.sqrt(self.expansion),
            activation_fn=self.activation_fn,
        )
        # TODO(antonin): check the init scale here
        self.alpha_scaler = Scaler(self.hidden_dim, init_scale=self.alpha_init, scale=self.alpha_scale)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.hyper_mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        return l2normalize(x, axis=-1)


class SimbaV2SquashedGaussianActor(nn.Module):
    # Note: each element in net_arch correpond to a residual block
    # not just a single layer
    net_arch: Sequence[int]
    action_dim: int
    # num_blocks: int = 2
    log_std_min: float = -20  # Note: -10 in SimBaV2 code
    log_std_max: float = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    scale_factor: int = 4
    constant_shift: float = 3.0

    def __post_init__(self):
        num_blocks = len(self.net_arch)
        assert num_blocks > 0, "SimbaV2 needs (num_blocks = len(net_arch)) > 0"
        hidden_dim = self.net_arch[0]
        self.scaler_init = np.sqrt(2.0 / hidden_dim).item()
        self.scaler_scale = np.sqrt(2.0 / hidden_dim).item()
        self.alpha_init = 1.0 / (num_blocks + 1.0)
        self.alpha_scale = 1.0 / np.sqrt(hidden_dim).item()
        super().__post_init__()

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = Flatten()(x)

        # Note: simba is using kernel_init=orthogonal_init(1)
        x = HyperEmbedder(
            hidden_dim=self.net_arch[0],
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            constant_shift=self.constant_shift,
        )(x)

        for n_units in self.net_arch:
            x = HyperLERPBlock(
                hidden_dim=n_units,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                activation_fn=self.activation_fn,
                expansion=self.scale_factor,
            )(x)

        mean_tmp = HyperDense(self.net_arch[-1])(x)
        mean_tmp = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(mean_tmp)
        mean = HyperDense(self.action_dim, use_bias=True)(mean_tmp)

        log_tmp = HyperDense(self.net_arch[-1])(x)
        log_tmp = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(log_tmp)
        log_std = HyperDense(self.action_dim, use_bias=True)(log_tmp)

        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        dist = TanhTransformedDistribution(
            tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std)),
        )
        return dist


class SimbaV2ContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False  # for consistency, not used
    dropout_rate: float | None = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    output_dim: int = 1
    scale_factor: int = 4
    constant_shift: float = 3.0

    def __post_init__(self):
        num_blocks = len(self.net_arch)
        assert num_blocks > 0, "SimbaV2 needs (num_blocks = len(net_arch)) > 0"
        hidden_dim = self.net_arch[0]
        self.scaler_init = np.sqrt(2.0 / hidden_dim).item()
        self.scaler_scale = np.sqrt(2.0 / hidden_dim).item()
        self.alpha_init = 1.0 / (num_blocks + 1.0)
        self.alpha_scale = 1.0 / np.sqrt(hidden_dim).item()
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = Flatten()(x)
        x = jnp.concatenate([x, action], -1)
        # Note: simba is using kernel_init=orthogonal_init(1)
        x = HyperEmbedder(
            hidden_dim=self.net_arch[0],
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            constant_shift=self.constant_shift,
        )(x)

        for n_units in self.net_arch:
            x = HyperLERPBlock(
                hidden_dim=n_units,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                activation_fn=self.activation_fn,
                expansion=self.scale_factor,
            )(x)
            # TODO: double check where to put the dropout
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        x = HyperDense(self.net_arch[-1])(x)
        x = Scaler(self.net_arch[-1], self.scaler_init, self.scaler_scale)(x)
        x = HyperDense(self.output_dim, use_bias=True)(x)

        return x


class SimbaV2VectorCritic(VectorCritic):
    # Note: we have use_layer_norm for consistency but it is not used
    # (other norm is applied)
    base_class: type[nn.Module] = SimbaV2ContinuousCritic
