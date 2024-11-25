from functools import partial
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Type, Union

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
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.tqc.policies import SimbaTQCPolicy, TQCPolicy


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


class TQC(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[TQCPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": TQCPolicy,
        "SimbaPolicy": SimbaTQCPolicy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": TQCPolicy,
    }

    policy: TQCPolicy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 1,
        top_quantiles_to_drop_per_net: int = 2,
        resets: Optional[List[int]] = None,  # List of timesteps after which to reset the params
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[Literal["auto"], float] = "auto",
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
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.ent_coef_init = ent_coef
        self.target_entropy = target_entropy
        self.resets = resets
        self.reset_idx = 0

        self.policy_kwargs["top_quantiles_to_drop_per_net"] = top_quantiles_to_drop_per_net

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
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_init, float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_init}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)  # type: ignore[assignment]

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.adam(
                    learning_rate=self.learning_rate,
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
        tb_log_name: str = "TQC",
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

        # Maybe reset the parameters
        if self.resets and self.reset_idx < len(self.resets) and self.num_timesteps >= self.resets[self.reset_idx]:
            # Note: we are not resetting the entropy coeff
            assert isinstance(self.qf_learning_rate, float)
            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)
            self.reset_idx += 1

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())  # type: ignore[attr-defined]
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(  # type: ignore[assignment]
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )
        (
            self.policy.qf1_state,
            self.policy.qf2_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
            (qf1_loss_value, qf2_loss_value, actor_loss_value, ent_coef_value),
        ) = self._train(
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            self.policy.n_target_quantiles,
            data,
            self.policy_delay,
            (self._n_updates + 1) % self.policy_delay,
            self.policy.qf1_state,
            self.policy.qf2_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.key,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf1_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())

    @staticmethod
    @partial(jax.jit, static_argnames=["n_target_quantiles"])
    def update_critic(
        gamma: float,
        n_target_quantiles: int,
        actor_state: TrainState,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        key: jax.Array,
    ):
        key, noise_key, dropout_key_1, dropout_key_2 = jax.random.split(key, 4)
        key, dropout_key_3, dropout_key_4 = jax.random.split(key, 3)
        # sample action from the actor
        dist = actor_state.apply_fn(actor_state.params, next_observations)
        next_state_actions = dist.sample(seed=noise_key)
        next_log_prob = dist.log_prob(next_state_actions)

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        qf1_next_quantiles = qf1_state.apply_fn(
            qf1_state.target_params,
            next_observations,
            next_state_actions,
            rngs={"dropout": dropout_key_1},
        )
        qf2_next_quantiles = qf1_state.apply_fn(
            qf2_state.target_params,
            next_observations,
            next_state_actions,
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

        def huber_quantile_loss(params: flax.core.FrozenDict, dropout_key: jax.Array) -> jax.Array:
            # Compute huber quantile loss
            current_quantiles = qf1_state.apply_fn(params, observations, actions, rngs={"dropout": dropout_key})
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
    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        key: jax.Array,
    ):
        key, dropout_key_1, dropout_key_2, noise_key = jax.random.split(key, 4)

        def actor_loss(params: flax.core.FrozenDict) -> Tuple[jax.Array, jax.Array]:
            dist = actor_state.apply_fn(params, observations)
            actor_actions = dist.sample(seed=noise_key)
            log_prob = dist.log_prob(actor_actions).reshape(-1, 1)

            qf1_pi = qf1_state.apply_fn(
                qf1_state.params,
                observations,
                actor_actions,
                rngs={"dropout": dropout_key_1},
            )
            qf2_pi = qf1_state.apply_fn(
                qf2_state.params,
                observations,
                actor_actions,
                rngs={"dropout": dropout_key_2},
            )
            qf1_pi = jnp.expand_dims(qf1_pi, axis=-1)
            qf2_pi = jnp.expand_dims(qf2_pi, axis=-1)

            # Concatenate quantiles from both critics
            # (batch, n_quantiles, n_critics)
            qf_pi = jnp.concatenate((qf1_pi, qf2_pi), axis=1)
            qf_pi = qf_pi.mean(axis=2).mean(axis=1, keepdims=True)

            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            return (ent_coef_value * log_prob - qf_pi).mean(), -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, (qf1_state, qf2_state), actor_loss_value, key, entropy

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf1_state: RLTrainState, qf2_state: RLTrainState) -> Tuple[RLTrainState, RLTrainState]:
        qf1_state = qf1_state.replace(target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, tau))
        qf2_state = qf2_state.replace(target_params=optax.incremental_update(qf2_state.params, qf2_state.target_params, tau))
        return qf1_state, qf2_state

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
        actor_state: RLTrainState,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        ent_coef_state: TrainState,
        observations: jax.Array,
        target_entropy: ArrayLike,
        key: jax.Array,
    ):
        (actor_state, (qf1_state, qf2_state), actor_loss_value, key, entropy) = cls.update_actor(
            actor_state,
            qf1_state,
            qf2_state,
            ent_coef_state,
            observations,
            key,
        )
        ent_coef_state, ent_coef_loss_value = cls.update_temperature(target_entropy, ent_coef_state, entropy)
        return actor_state, (qf1_state, qf2_state), ent_coef_state, actor_loss_value, ent_coef_loss_value, key

    @classmethod
    @partial(
        jax.jit,
        static_argnames=["cls", "gradient_steps", "n_target_quantiles", "policy_delay", "policy_delay_offset"],
    )
    def _train(
        cls,
        gamma: float,
        tau: float,
        target_entropy: ArrayLike,
        gradient_steps: int,
        n_target_quantiles: int,
        data: ReplayBufferSamplesNp,
        policy_delay: int,
        policy_delay_offset: int,
        qf1_state: RLTrainState,
        qf2_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        key: jax.Array,
    ):
        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // gradient_steps

        carry = {
            "actor_state": actor_state,
            "qf1_state": qf1_state,
            "qf2_state": qf2_state,
            "ent_coef_state": ent_coef_state,
            "key": key,
            "info": {
                "actor_loss": jnp.array(0.0),
                "qf1_loss": jnp.array(0.0),
                "qf2_loss": jnp.array(0.0),
                "ent_coef_loss": jnp.array(0.0),
            },
        }

        def one_update(i: int, carry: Dict[str, Any]) -> Dict[str, Any]:
            # Note: this method must be defined inline because
            # `fori_loop` expect a signature fn(index, carry) -> carry
            actor_state = carry["actor_state"]
            qf1_state = carry["qf1_state"]
            qf2_state = carry["qf2_state"]
            ent_coef_state = carry["ent_coef_state"]
            key = carry["key"]
            info = carry["info"]
            batch_obs = jax.lax.dynamic_slice_in_dim(data.observations, i * batch_size, batch_size)
            batch_act = jax.lax.dynamic_slice_in_dim(data.actions, i * batch_size, batch_size)
            batch_next_obs = jax.lax.dynamic_slice_in_dim(data.next_observations, i * batch_size, batch_size)
            batch_rew = jax.lax.dynamic_slice_in_dim(data.rewards, i * batch_size, batch_size)
            batch_done = jax.lax.dynamic_slice_in_dim(data.dones, i * batch_size, batch_size)
            (
                (qf1_state, qf2_state),
                (qf1_loss_value, qf2_loss_value, ent_coef_value),
                key,
            ) = cls.update_critic(
                gamma,
                n_target_quantiles,
                actor_state,
                qf1_state,
                qf2_state,
                ent_coef_state,
                batch_obs,
                batch_act,
                batch_next_obs,
                batch_rew,
                batch_done,
                key,
            )
            qf1_state, qf2_state = cls.soft_update(tau, qf1_state, qf2_state)

            (actor_state, (qf1_state, qf2_state), ent_coef_state, actor_loss_value, ent_coef_loss_value, key) = jax.lax.cond(
                (policy_delay_offset + i) % policy_delay == 0,
                # If True:
                cls.update_actor_and_temperature,
                # If False:
                lambda *_: (
                    actor_state,
                    (qf1_state, qf2_state),
                    ent_coef_state,
                    info["actor_loss"],
                    info["ent_coef_loss"],
                    key,
                ),
                actor_state,
                qf1_state,
                qf2_state,
                ent_coef_state,
                batch_obs,
                target_entropy,
                key,
            )
            info = {
                "actor_loss": actor_loss_value,
                "qf1_loss": qf1_loss_value,
                "qf2_loss": qf2_loss_value,
                "ent_coef_loss": ent_coef_loss_value,
            }

            return {
                "actor_state": actor_state,
                "qf1_state": qf1_state,
                "qf2_state": qf2_state,
                "ent_coef_state": ent_coef_state,
                "key": key,
                "info": info,
            }

        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        return (
            update_carry["qf1_state"],
            update_carry["qf2_state"],
            update_carry["actor_state"],
            update_carry["ent_coef_state"],
            update_carry["key"],
            (
                update_carry["info"]["qf1_loss"],
                update_carry["info"]["qf2_loss"],
                update_carry["info"]["actor_loss"],
                update_carry["info"]["ent_coef_loss"],
            ),
        )
