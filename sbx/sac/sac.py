from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.sac.policies import SACPolicy


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        # TODO: add parameter in train to remove that hack
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init


class SAC(OffPolicyAlgorithmJax):

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
        gradient_steps: int = 1,
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
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(gym.spaces.Box),
            support_multi_env=True,
        )

        self.ent_coef_init = ent_coef

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.policy is None:
            self.policy = self.policy_class(  # pytype:disable=not-instantiable
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,  # pytype:disable=not-instantiable
            )

            self.key = self.policy.build(self.key, self.lr_schedule)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor
            self.qf = self.policy.qf

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith("auto"):
                # Default initial value of ent_coef when learned
                ent_coef_init = 1.0
                if "_" in self.ent_coef_init:
                    ent_coef_init = float(self.ent_coef_init.split("_")[1])
                    assert ent_coef_init > 0.0, "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.ent_coef = EntropyCoef(ent_coef_init)
            else:
                # Force conversion to float
                # this will throw an error if a malformed string (different from 'auto')
                # is passed
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.adam(
                    learning_rate=self.learning_rate,
                ),
            )

        # automatically set target entropy if needed
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "TQC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps)
        n_updates = 0
        # Convert to numpy
        data = ReplayBufferSamplesNp(
            data.observations.numpy(),
            data.actions.numpy(),
            data.next_observations.numpy(),
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            n_updates,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            (qf_loss_value, actor_loss_value),
        ) = self._train(
            self.actor,
            self.qf,
            self.ent_coef,
            self.gamma,
            self.tau,
            self.target_entropy,
            self.gradient_steps,
            data,
            n_updates,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["actor", "qf", "ent_coef", "gamma"])
    def update_critic(
        actor,
        qf,
        ent_coef,
        gamma,
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key: jnp.ndarray,
    ):
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
        key, noise_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)
        # sample action from the actor
        dist = actor.apply(actor_state.params, next_observations)
        next_state_actions = dist.sample(seed=noise_key)
        next_log_prob = dist.log_prob(next_state_actions)

        ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})

        qf_next_values = qf.apply(
            qf_state.target_params,
            next_observations,
            next_state_actions,
            rngs={"dropout": dropout_key_target},
        )

        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values

        def mse_loss(params, dropout_key):
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf.apply(params, observations, actions, rngs={"dropout": dropout_key})
            return ((target_q_values - current_q_values) ** 2).mean()

        qf_loss_value, grads = jax.value_and_grad(mse_loss, has_aux=False)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            (qf_state, ent_coef_state),
            qf_loss_value,
            key,
        )

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
            # Take min among all critics (mean for droq)
            min_qf_pi = jnp.min(qf_pi, axis=1, keepdims=True)
            ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})

            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy

    @staticmethod
    @partial(jax.jit, static_argnames=["tau"])
    def soft_update(tau, qf_state: RLTrainState):
        qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        return qf_state

    @staticmethod
    @partial(jax.jit, static_argnames=["ent_coef", "target_entropy"])
    def update_temperature(ent_coef, target_entropy, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params):
            ent_coef_value = ent_coef.apply({"params": temp_params})
            # ent_coef_loss = (jnp.log(ent_coef_value) * (entropy - target_entropy)).mean()
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

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
        data: ReplayBufferSamplesNp,
        n_updates: int,
        qf_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        key,
    ):
        for i in range(gradient_steps):
            n_updates += 1

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                # batch_size = batch_size
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]

            ((qf_state, ent_coef_state), qf_loss_value, key,) = SAC.update_critic(
                actor,
                qf,
                ent_coef,
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                key,
            )
            qf_state = SAC.soft_update(tau, qf_state)

            (actor_state, qf_state, actor_loss_value, key, entropy) = SAC.update_actor(
                actor,
                qf,
                ent_coef,
                actor_state,
                qf_state,
                ent_coef_state,
                slice(data.observations),
                key,
            )
            ent_coef_state, _ = SAC.update_temperature(ent_coef, target_entropy, ent_coef_state, entropy)

        return (
            n_updates,
            qf_state,
            actor_state,
            ent_coef_state,
            key,
            (qf_loss_value, actor_loss_value),
        )
