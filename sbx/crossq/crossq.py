from functools import partial
from typing import Any, ClassVar, Literal, Optional, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax.typing import ArrayLike
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import BatchNormTrainState, ReplayBufferSamplesNp
from sbx.crossq.policies import CrossQPolicy, SimbaCrossQPolicy


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        # Hack to not optimize the entropy coefficient while not having to use if/else for the jit
        # TODO: add parameter in train to remove that hack
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init


class CrossQ(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[dict[str, type[CrossQPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": CrossQPolicy,
        "SimbaPolicy": SimbaCrossQPolicy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": CrossQPolicy,
    }

    policy: CrossQPolicy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 3,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[Literal["auto"], float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        param_resets: Optional[list[int]] = None,  # List of timesteps after which to reset the params
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            n_steps=n_steps,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
            policy_kwargs=policy_kwargs,
            param_resets=param_resets,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.ent_coef_init = ent_coef
        self.target_entropy = target_entropy

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]

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
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_init, float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_init}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)  # type: ignore[assignment]

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.adam(
                    learning_rate=self.lr_schedule(1),
                ),
            )

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "CrossQ",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)

        # Maybe reset the parameters/optimizers fully
        self._maybe_reset_params()

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())  # type: ignore[attr-defined]
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        if data.discounts is None:
            discounts = np.full((batch_size * gradient_steps,), self.gamma, dtype=np.float32)
        else:
            # For bootstrapping with n-step returns
            discounts = data.discounts.numpy().flatten()

        # Convert to numpy
        data = ReplayBufferSamplesNp(  # type: ignore[assignment]
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
            discounts,
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            (actor_loss_value, qf_loss_value, ent_coef_loss_value, ent_coef_value),
        ) = self._train(
            self.target_entropy,
            gradient_steps,
            data,
            self.policy_delay,
            (self._n_updates + 1) % self.policy_delay,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/ent_coef_loss", ent_coef_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())

    @staticmethod
    @jax.jit
    def update_critic(
        actor_state: BatchNormTrainState,
        qf_state: BatchNormTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        discounts: jax.Array,
        key: jax.Array,
    ):
        key, noise_key, dropout_key_current = jax.random.split(key, 3)
        # sample action from the actor
        dist = actor_state.apply_fn(
            {"params": actor_state.params, "batch_stats": actor_state.batch_stats},
            next_observations,
            train=False,
        )
        next_state_actions = dist.sample(seed=noise_key)
        next_log_prob = dist.log_prob(next_state_actions)

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        def mse_loss(
            params: flax.core.FrozenDict, batch_stats: flax.core.FrozenDict, dropout_key: flax.core.FrozenDict
        ) -> tuple[jax.Array, jax.Array]:
            # Joint forward pass of obs/next_obs and actions/next_state_actions to have only
            # one forward pass with shape (n_critics, 2 * batch_size, 1).
            #
            # This has two reasons:
            # 1. According to the paper obs/actions and next_obs/next_state_actions are differently
            #    distributed which is the reason why "naively" applying Batch Normalization in SAC fails.
            #    The batch statistics have to instead be calculated for the mixture distribution of obs/next_obs
            #    and actions/next_state_actions. Otherwise, next_obs/next_state_actions are perceived as
            #    out-of-distribution to the Batch Normalization layer, since running statistics are only polyak averaged
            #    over from the live network and have never seen the next batch which is known to be unstable.
            #    Without target networks, the joint forward pass is a simple solution to calculate
            #    the joint batch statistics directly with a single forward pass.
            #
            # 2. From a computational perspective a single forward pass is simply more efficient than
            #    two sequential forward passes.
            q_values, state_updates = qf_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                jnp.concatenate([observations, next_observations], axis=0),
                jnp.concatenate([actions, next_state_actions], axis=0),
                rngs={"dropout": dropout_key},
                mutable=["batch_stats"],
                train=True,
            )

            # Recover current and next_q_values, split into two
            current_q_values, qf_next_values = jnp.split(q_values, 2, axis=1)

            # Compute target q_values
            next_q_values = jnp.min(qf_next_values, axis=0)
            # td error + entropy term
            next_q_values = next_q_values - ent_coef_value * next_log_prob[:, None]
            # shape is (batch_size, 1)
            target_q_values = rewards[:, None] + (1 - dones[:, None]) * discounts[:, None] * next_q_values

            return 0.5 * ((jax.lax.stop_gradient(target_q_values) - current_q_values) ** 2).mean(axis=1).sum(), state_updates

        (qf_loss_value, state_updates), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params, qf_state.batch_stats, dropout_key_current
        )
        qf_state = qf_state.apply_gradients(grads=grads)
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])

        return (
            qf_state,
            (qf_loss_value, ent_coef_value),
            key,
        )

    @staticmethod
    @jax.jit
    def update_actor(
        actor_state: BatchNormTrainState,
        qf_state: BatchNormTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
    ):
        key, dropout_key, noise_key = jax.random.split(key, 3)

        def actor_loss(
            params: flax.core.FrozenDict, batch_stats: flax.core.FrozenDict
        ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            dist, state_updates = actor_state.apply_fn(
                {"params": params, "batch_stats": batch_stats},
                observations,
                mutable=["batch_stats"],
                train=True,
            )
            actor_actions = dist.sample(seed=noise_key)
            log_prob = dist.log_prob(actor_actions).reshape(-1, 1)

            qf_pi = qf_state.apply_fn(
                {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
                observations,
                actor_actions,
                rngs={"dropout": dropout_key},
                train=False,
            )
            # Take min among all critics (mean for droq)
            min_qf_pi = jnp.min(qf_pi, axis=0)
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, (-log_prob.mean(), state_updates)

        (actor_loss_value, (entropy, state_updates)), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params, actor_state.batch_stats
        )
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(batch_stats=state_updates["batch_stats"])

        return actor_state, qf_state, actor_loss_value, key, entropy

    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: ArrayLike, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params: flax.core.FrozenDict) -> jax.Array:
            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = jnp.log(ent_coef_value) * (entropy - target_entropy).mean()  # type: ignore[union-attr]
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @classmethod
    def update_actor_and_temperature(
        cls,
        actor_state: BatchNormTrainState,
        qf_state: BatchNormTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        target_entropy: ArrayLike,
        key: jax.Array,
    ):
        (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
            actor_state,
            qf_state,
            ent_coef_state,
            observations,
            key,
        )
        ent_coef_state, ent_coef_loss_value = cls.update_temperature(target_entropy, ent_coef_state, entropy)
        return actor_state, qf_state, ent_coef_state, actor_loss_value, ent_coef_loss_value, key

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps", "policy_delay", "policy_delay_offset"])
    def _train(
        cls,
        target_entropy: ArrayLike,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay: int,
        policy_delay_offset: int,
        qf_state: BatchNormTrainState,
        actor_state: BatchNormTrainState,
        ent_coef_state: TrainState,
        key: jax.Array,
    ):
        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // gradient_steps

        carry = {
            "actor_state": actor_state,
            "qf_state": qf_state,
            "ent_coef_state": ent_coef_state,
            "key": key,
            "info": {
                "actor_loss": jnp.array(0.0),
                "qf_loss": jnp.array(0.0),
                "ent_coef_loss": jnp.array(0.0),
                "ent_coef_value": jnp.array(0.0),
            },
        }

        def one_update(i: int, carry: dict[str, Any]) -> dict[str, Any]:
            # Note: this method must be defined inline because
            # `fori_loop` expect a signature fn(index, carry) -> carry
            actor_state = carry["actor_state"]
            qf_state = carry["qf_state"]
            ent_coef_state = carry["ent_coef_state"]
            key = carry["key"]
            info = carry["info"]
            batch_obs = jax.lax.dynamic_slice_in_dim(data.observations, i * batch_size, batch_size)
            batch_actions = jax.lax.dynamic_slice_in_dim(data.actions, i * batch_size, batch_size)
            batch_next_obs = jax.lax.dynamic_slice_in_dim(data.next_observations, i * batch_size, batch_size)
            batch_rewards = jax.lax.dynamic_slice_in_dim(data.rewards, i * batch_size, batch_size)
            batch_dones = jax.lax.dynamic_slice_in_dim(data.dones, i * batch_size, batch_size)
            batch_discounts = jax.lax.dynamic_slice_in_dim(data.discounts, i * batch_size, batch_size)
            (
                qf_state,
                (qf_loss_value, ent_coef_value),
                key,
            ) = cls.update_critic(
                actor_state,
                qf_state,
                ent_coef_state,
                batch_obs,
                batch_actions,
                batch_next_obs,
                batch_rewards,
                batch_dones,
                batch_discounts,
                key,
            )
            # No target q values with CrossQ

            (actor_state, qf_state, ent_coef_state, actor_loss_value, ent_coef_loss_value, key) = jax.lax.cond(
                (policy_delay_offset + i) % policy_delay == 0,
                # If True:
                cls.update_actor_and_temperature,
                # If False:
                lambda *_: (actor_state, qf_state, ent_coef_state, info["actor_loss"], info["ent_coef_loss"], key),
                actor_state,
                qf_state,
                ent_coef_state,
                batch_obs,
                target_entropy,
                key,
            )
            info = {
                "actor_loss": actor_loss_value,
                "qf_loss": qf_loss_value,
                "ent_coef_loss": ent_coef_loss_value,
                "ent_coef_value": ent_coef_value,
            }

            return {
                "actor_state": actor_state,
                "qf_state": qf_state,
                "ent_coef_state": ent_coef_state,
                "key": key,
                "info": info,
            }

        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        return (
            update_carry["qf_state"],
            update_carry["actor_state"],
            update_carry["ent_coef_state"],
            update_carry["key"],
            (
                update_carry["info"]["actor_loss"],
                update_carry["info"]["qf_loss"],
                update_carry["info"]["ent_coef_loss"],
                update_carry["info"]["ent_coef_value"],
            ),
        )
