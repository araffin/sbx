from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
from flax.training.train_state import TrainState
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.tqc.policies import TQCPolicy
from sbx.tqc.tqc import TQC


class DroQ(TQC):

    policy_aliases: Dict[str, Optional[nn.Module]] = {
        "MlpPolicy": TQCPolicy,
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
        gradient_steps: int = 1,
        top_quantiles_to_drop_per_net: int = 2,
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
            action_noise=action_noise,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            top_quantiles_to_drop_per_net=top_quantiles_to_drop_per_net,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )
        self.policy_kwargs["dropout_rate"] = dropout_rate
        self.policy_kwargs["layer_norm"] = layer_norm

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "actor",
            "qf",
            "ent_coef",
            "gamma",
            "tau",
            "target_entropy",
            "gradient_steps",
            "n_target_quantiles",
        ],
    )
    def _train(
        actor,
        qf,
        ent_coef,
        gamma,
        tau,
        target_entropy,
        gradient_steps,
        n_target_quantiles,
        data: ReplayBufferSamplesNp,
        n_updates: int,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        key,
    ):
        for i in range(gradient_steps):
            n_updates += 1

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]

            ((qf1_state, qf2_state), (qf1_loss_value, qf2_loss_value, ent_coef_value), key,) = TQC.update_critic(
                actor,
                qf,
                ent_coef,
                gamma,
                n_target_quantiles,
                actor_state,
                qf1_state,
                qf2_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                key,
            )
            qf1_state, qf2_state = TQC.soft_update(tau, qf1_state, qf2_state)

        # DroQ only updates the actor once
        (actor_state, (qf1_state, qf2_state), actor_loss_value, key, entropy) = TQC.update_actor(
            actor,
            qf,
            ent_coef,
            actor_state,
            qf1_state,
            qf2_state,
            ent_coef_state,
            slice(data.observations),
            key,
        )
        ent_coef_state, _ = TQC.update_temperature(ent_coef, target_entropy, ent_coef_state, entropy)

        return (
            n_updates,
            qf1_state,
            qf2_state,
            actor_state,
            ent_coef_state,
            key,
            (qf1_loss_value, qf2_loss_value, actor_loss_value, ent_coef_value),
        )
