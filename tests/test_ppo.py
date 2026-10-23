from typing import Optional, Any

import flax.linen as nn
import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env
import optax
import jax
from gymnasium import spaces
from flax.training.train_state import TrainState
import jax.numpy as jnp

from sbx import PPO

from sbx.ppo.policies import PPOPolicy, Actor, Critic



class CustomPPO(PPOPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch = None, ortho_init = False, log_std_init = 0, activation_fn = nn.tanh, use_sde = False, use_expln = False, clip_mean = 2, features_extractor_class=None, features_extractor_kwargs = None, normalize_images = True, optimizer_class = optax.adam, optimizer_kwargs = None, share_features_extractor = False, actor_class = Actor, critic_class = Critic):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, ortho_init, log_std_init, activation_fn, use_sde, use_expln, clip_mean, features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class, optimizer_kwargs, share_features_extractor, actor_class, critic_class)

    
    def build(self, key: jax.Array, lr_schedule, max_grad_norm) -> jax.Array:

        # Coustom PPO Policy build  

        key, actor_key, vf_key = jax.random.split(key, 3)
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample()])

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs: dict[str, Any] = {
                "action_dim": int(np.prod(self.action_space.shape)),
            }
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            }
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            assert self.action_space.nvec.ndim == 1, (
                "Only one-dimensional MultiDiscrete action spaces are supported, "
                f"but found MultiDiscrete({(self.action_space.nvec).tolist()})."
            )
            actor_kwargs = {
                "action_dim": int(np.sum(self.action_space.nvec)),
                "num_discrete_choices": self.action_space.nvec,  # type: ignore[dict-item]
            }
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            # Handle binary action spaces as discrete action spaces with two choices.
            actor_kwargs = {
                "action_dim": 2 * self.action_space.n,
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            }
        else:
            raise NotImplementedError(f"{self.action_space}")

        self.actor = self.actor_class(
            net_arch=self.net_arch_pi,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            ortho_init=self.ortho_init,
            **actor_kwargs,  # type: ignore[arg-type]
        )

        self.actor.reset_noise = self.reset_noise
        optimizer_class = optax.inject_hyperparams(self.optimizer_class)(learning_rate=lr_schedule(1), **self.optimizer_kwargs)

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                # optax.clip_by_global_norm(max_grad_norm), # ->  Test an Optax chain with only one element.
                optimizer_class,
            ),
        )

        self.vf = self.critic_class(net_arch=self.net_arch_vf, activation_fn=self.activation_fn)

        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init(vf_key, obs),
            tx=optax.chain(
                # optax.clip_by_global_norm(max_grad_norm), # -> Test an Optax chain with only one element.
                optimizer_class,
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]

        return key 

def test_ppo() -> None:
    env = make_vec_env('Pendulum-v1')

    # PPO assumes that the train state (self.vf_state) is created from an Optax chain with two elements.
    model = PPO(
        CustomPPO,
        env
    )
    
    model.learn(64, progress_bar=True)
