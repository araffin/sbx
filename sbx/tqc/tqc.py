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
from sbx.tqc.policies import TQCPolicy


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


class TQC(OffPolicyAlgorithmJax):

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
        self.policy_kwargs["top_quantiles_to_drop_per_net"] = top_quantiles_to_drop_per_net

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
        # Convert to numpy
        data = ReplayBufferSamplesNp(
            data.observations.numpy(),
            data.actions.numpy(),
            data.next_observations.numpy(),
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self._n_updates,
            self.policy.qf1_state,
            self.policy.qf2_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            (qf1_loss_value, qf2_loss_value, actor_loss_value, ent_coef_value),
        ) = self._train(
            self.actor,
            self.qf,
            self.ent_coef,
            self.gamma,
            self.tau,
            self.target_entropy,
            self.gradient_steps,
            self.policy.n_target_quantiles,
            data,
            self._n_updates,
            self.policy.qf1_state,
            self.policy.qf2_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
        )
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf1_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())

    @staticmethod
    @partial(jax.jit, static_argnames=["actor", "qf", "ent_coef", "gamma", "n_target_quantiles"])
    def update_critic(
        actor,
        qf,
        ent_coef,
        gamma,
        n_target_quantiles,
        actor_state: TrainState,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key: jnp.ndarray,
    ):
        key, noise_key, dropout_key_1, dropout_key_2 = jax.random.split(key, 4)
        key, dropout_key_3, dropout_key_4 = jax.random.split(key, 3)
        # sample action from the actor
        dist = actor.apply(actor_state.params, next_observations)
        next_state_actions = dist.sample(seed=noise_key)
        next_log_prob = dist.log_prob(next_state_actions)

        ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})

        qf1_next_quantiles = qf.apply(
            qf1_state.target_params,
            next_observations,
            next_state_actions,
            True,
            rngs={"dropout": dropout_key_1},
        )
        qf2_next_quantiles = qf.apply(
            qf2_state.target_params,
            next_observations,
            next_state_actions,
            True,
            rngs={"dropout": dropout_key_2},
        )

        # Concatenate quantiles from both critics to get a single tensor
        # batch x quantiles
        qf_next_quantiles = jnp.concatenate((qf1_next_quantiles, qf2_next_quantiles), axis=1)

        # sort next quantiles with jax
        next_quantiles = jnp.sort(qf_next_quantiles)
        # Keep only the quantiles we need
        next_target_quantiles = next_quantiles[:, :n_target_quantiles]

        # td error + entropy term
        next_target_quantiles = next_target_quantiles - ent_coef_value * next_log_prob.reshape(-1, 1)
        target_quantiles = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_target_quantiles

        # Make target_quantiles broadcastable to (batch_size, n_quantiles, n_target_quantiles).
        target_quantiles = jnp.expand_dims(target_quantiles, axis=1)

        def huber_quantile_loss(params, noise_key):
            # Compute huber quantile loss
            current_quantiles = qf.apply(params, observations, actions, True, rngs={"dropout": noise_key})
            # convert to shape: (batch_size, n_quantiles, 1) for broadcast
            current_quantiles = jnp.expand_dims(current_quantiles, axis=-1)

            n_quantiles = current_quantiles.shape[1]
            # Cumulative probabilities to calculate quantiles.
            # shape: (n_quantiles,)
            cum_prob = (jnp.arange(n_quantiles, dtype=jnp.float32) + 0.5) / n_quantiles
            # convert to shape: (1, n_quantiles, 1) for broadcast
            cum_prob = jnp.expand_dims(cum_prob, axis=(0, -1))

            # pairwise_delta: (batch_size, n_quantiles, n_target_quantiles)
            pairwise_delta = target_quantiles - current_quantiles
            abs_pairwise_delta = jnp.abs(pairwise_delta)
            huber_loss = jnp.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
            loss = jnp.abs(cum_prob - (pairwise_delta < 0).astype(jnp.float32)) * huber_loss
            return loss.mean()

        qf1_loss_value, grads1 = jax.value_and_grad(huber_quantile_loss, has_aux=False)(qf1_state.params, dropout_key_3)
        qf2_loss_value, grads2 = jax.value_and_grad(huber_quantile_loss, has_aux=False)(qf2_state.params, dropout_key_4)
        qf1_state = qf1_state.apply_gradients(grads=grads1)
        qf2_state = qf2_state.apply_gradients(grads=grads2)

        return (
            (qf1_state, qf2_state),
            (qf1_loss_value, qf2_loss_value, ent_coef_value),
            key,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["actor", "qf", "ent_coef"])
    def update_actor(
        actor,
        qf,
        ent_coef,
        actor_state: RLTrainState,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: np.ndarray,
        key: jnp.ndarray,
    ):
        key, dropout_key_1, dropout_key_2, noise_key = jax.random.split(key, 4)

        def actor_loss(params):

            dist = actor.apply(params, observations)
            actor_actions = dist.sample(seed=noise_key)
            log_prob = dist.log_prob(actor_actions).reshape(-1, 1)

            qf1_pi = qf.apply(
                qf1_state.params,
                observations,
                actor_actions,
                True,
                rngs={"dropout": dropout_key_1},
            )
            qf2_pi = qf.apply(
                qf2_state.params,
                observations,
                actor_actions,
                True,
                rngs={"dropout": dropout_key_2},
            )
            qf1_pi = jnp.expand_dims(qf1_pi, axis=-1)
            qf2_pi = jnp.expand_dims(qf2_pi, axis=-1)

            # Concatenate quantiles from both critics
            # (batch, n_quantiles, n_critics)
            qf_pi = jnp.concatenate((qf1_pi, qf2_pi), axis=1)
            qf_pi = qf_pi.mean(axis=2).mean(axis=1, keepdims=True)

            ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})
            return (ent_coef_value * log_prob - qf_pi).mean(), -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, (qf1_state, qf2_state), actor_loss_value, key, entropy

    @staticmethod
    @partial(jax.jit, static_argnames=["tau"])
    def soft_update(tau, qf1_state: RLTrainState, qf2_state: RLTrainState):
        qf1_state = qf1_state.replace(target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, tau))
        qf2_state = qf2_state.replace(target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, tau))
        return qf1_state, qf2_state

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
