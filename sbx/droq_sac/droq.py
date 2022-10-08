from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from sbx.common.type_aliases import RLTrainState
from sbx.sac.policies import SACPolicy
from sbx.sac.sac import SAC


class DroQ_SAC(SAC):

    policy_aliases: Dict[str, Optional[nn.Module]] = {
        "MlpPolicy": SACPolicy,
    }

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 2,
        # policy_delay = gradient_steps to follow original implementation
        policy_delay: int = 2,
        dropout_rate: float = 0.01,
        layer_norm: bool = True,
        action_noise: Optional[ActionNoise] = None,
        ent_coef: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_delay=policy_delay,
            action_noise=action_noise,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )
        self.policy_kwargs["dropout_rate"] = dropout_rate
        self.policy_kwargs["layer_norm"] = layer_norm

        if _init_setup_model:
            self._setup_model()

    @staticmethod
    @partial(jax.jit, static_argnames=["actor", "qf", "ent_coef"])
    def update_actor(
        actor,
        qf,
        ent_coef,
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: np.ndarray,
        key: jnp.ndarray,
    ):
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def actor_loss(params):

            dist = actor.apply(params, observations)
            actor_actions = dist.sample(seed=noise_key)
            log_prob = dist.log_prob(actor_actions).reshape(-1, 1)

            qf_pi = qf.apply(
                qf_state.params,
                observations,
                actor_actions,
                rngs={"dropout": dropout_key},
            )
            # Take mean among all critics (min for SAC)
            min_qf_pi = jnp.mean(qf_pi, axis=0)
            ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy
