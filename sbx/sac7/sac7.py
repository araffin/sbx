from functools import partial
from typing import Any, Dict, Optional, Tuple, Type, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule

from sbx.common.off_policy_algorithm import OffPolicyAlgorithmJax
from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from sbx.sac7.policies import SAC7Policy


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


class SAC7(OffPolicyAlgorithmJax):
    """
    Including State-Action Representation learning in SAC
    TD7: https://github.com/sfujim/TD7

    Note: The rest of the tricks (LAP replay buffer, checkpoints, extrapolation error correction, huber loss)
    are not yet implemented
    """

    policy_aliases: Dict[str, Type[SAC7Policy]] = {  # type: ignore[assignment]
        "MlpPolicy": SAC7Policy,
        # Minimal dict support using flatten()
        "MultiInputPolicy": SAC7Policy,
    }

    policy: SAC7Policy
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
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
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

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            # pytype: disable=not-instantiable
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )
            # pytype: enable=not-instantiable

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]
            self.encoder = self.policy.encoder  # type: ignore[assignment]
            self.action_encoder = self.policy.action_encoder  # type: ignore[assignment]

            # Value clipping tracked values
            self.min_qf_target = 0
            self.max_qf_target = 0

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

        # automatically set target entropy if needed
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
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

    def train(self, batch_size, gradient_steps):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        # Pre-compute the indices where we need to update the actor
        # This is a hack in order to jit the train loop
        # It will compile once per value of policy_delay_indices
        policy_delay_indices = {i: True for i in range(gradient_steps) if ((self._n_updates + i + 1) % self.policy_delay) == 0}
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate([data.observations[key].numpy() for key in keys], axis=1)
            next_obs = np.concatenate([data.next_observations[key].numpy() for key in keys], axis=1)
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Initialize clipping at the first iteration
        if self._n_updates == 0:
            self.min_qf_target = data.rewards.min().item()
            self.max_qf_target = data.rewards.max().item()

        # Convert to numpy
        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.policy.encoder_state,
            self.policy.action_encoder_state,
            self.key,
            (actor_loss_value, qf_loss_value, ent_coef_value),
            (self.min_qf_target, self.max_qf_target),
        ) = self._train(
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.policy.encoder_state,
            self.policy.action_encoder_state,
            self.key,
            self.min_qf_target,
            self.max_qf_target,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/ent_coef", ent_coef_value.item())

    @staticmethod
    @jax.jit
    def update_critic(
        gamma: float,
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        encoder_state: RLTrainState,
        action_encoder_state: RLTrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        key: jax.random.KeyArray,
        min_qf_target: float,
        max_qf_target: float,
    ):
        key, noise_key, dropout_key_target, dropout_key_current = jax.random.split(key, 4)

        z_next = encoder_state.apply_fn(encoder_state.target_params, next_observations)

        # sample action from the actor
        dist = actor_state.apply_fn(actor_state.params, next_observations, z_next)
        next_state_actions = dist.sample(seed=noise_key)
        next_log_prob = dist.log_prob(next_state_actions)

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        z_action_next = action_encoder_state.apply_fn(action_encoder_state.target_params, z_next, next_state_actions)

        qf_next_values = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            next_state_actions,
            z_next,
            z_action_next,
            rngs={"dropout": dropout_key_target},
        )

        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)

        # Clip next q_values
        next_q_values = jnp.clip(next_q_values, min_qf_target, max_qf_target)

        # shape is (batch_size, 1)
        target_q_values = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values

        # Compute min/max to update clipping range
        # TODO: initialize them properly
        qf_min = jnp.minimum(target_q_values.min(), min_qf_target)
        qf_max = jnp.maximum(target_q_values.max(), max_qf_target)

        z_state = encoder_state.apply_fn(encoder_state.target_params, observations)
        z_state_action = action_encoder_state.apply_fn(action_encoder_state.target_params, z_state, actions)

        def mse_loss(params, dropout_key):
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf_state.apply_fn(
                params,
                observations,
                actions,
                z_state,
                z_state_action,
                rngs={"dropout": dropout_key},
            )
            return 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()

        qf_loss_value, grads = jax.value_and_grad(mse_loss, has_aux=False)(qf_state.params, dropout_key_current)
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, ent_coef_value),
            key,
            (qf_min, qf_max),
        )

    @staticmethod
    @jax.jit
    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        encoder_state: RLTrainState,
        action_encoder_state: RLTrainState,
        observations: np.ndarray,
        key: jax.random.KeyArray,
    ):
        key, dropout_key, noise_key = jax.random.split(key, 3)

        z_state = encoder_state.apply_fn(encoder_state.target_params, observations)

        def actor_loss(params):
            dist = actor_state.apply_fn(params, observations, z_state)
            actor_actions = dist.sample(seed=noise_key)
            log_prob = dist.log_prob(actor_actions).reshape(-1, 1)

            z_state_action = action_encoder_state.apply_fn(action_encoder_state.target_params, z_state, actor_actions)

            qf_pi = qf_state.apply_fn(
                qf_state.params,
                observations,
                actor_actions,
                z_state,
                z_state_action,
                rngs={"dropout": dropout_key},
            )
            # Take min among all critics (mean for droq)
            min_qf_pi = jnp.min(qf_pi, axis=0)
            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, qf_state, actor_loss_value, key, entropy

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState):
        qf_state = qf_state.replace(target_params=optax.incremental_update(qf_state.params, qf_state.target_params, tau))
        return qf_state

    @staticmethod
    @jax.jit
    def update_temperature(target_entropy: np.ndarray, ent_coef_state: TrainState, entropy: float):
        def temperature_loss(temp_params):
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = ent_coef_value * (entropy - target_entropy).mean()
            return ent_coef_loss

        ent_coef_loss, grads = jax.value_and_grad(temperature_loss)(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss

    @staticmethod
    @jax.jit
    def update_encoders(
        encoder_state: RLTrainState,
        action_encoder_state: RLTrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
    ):
        z_next_state = jax.lax.stop_gradient(encoder_state.apply_fn(encoder_state.params, next_observations))

        def encoder_loss(encoder_params, action_encoder_params):
            z_state = encoder_state.apply_fn(encoder_params, observations)
            z_state_action = action_encoder_state.apply_fn(action_encoder_params, z_state, actions)
            # encoder_loss =  optax.huber_loss(z_state_action, z_next_state).mean()
            return optax.l2_loss(z_state_action, z_next_state).mean()

        _, (encoder_grads, action_encoder_grads) = jax.value_and_grad(encoder_loss, argnums=(0, 1))(
            encoder_state.params,
            action_encoder_state.params,
        )

        encoder_state = encoder_state.apply_gradients(grads=encoder_grads)
        action_encoder_state = action_encoder_state.apply_gradients(grads=action_encoder_grads)

        return encoder_state, action_encoder_state

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps"])
    def _train(
        cls,
        gamma: float,
        tau: float,
        target_entropy: np.ndarray,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay_indices: flax.core.FrozenDict,
        qf_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        encoder_state: RLTrainState,
        action_encoder_state: RLTrainState,
        key: jax.random.KeyArray,
        min_qf_target: float,
        max_qf_target: float,
    ):
        actor_loss_value = jnp.array(0)

        for i in range(gradient_steps):

            def slice(x, step=i):
                assert x.shape[0] % gradient_steps == 0
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * step : batch_size * (step + 1)]

            encoder_state, action_encoder_state = SAC7.update_encoders(
                encoder_state,
                action_encoder_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
            )

            (
                qf_state,
                (qf_loss_value, ent_coef_value),
                key,
                (qf_min, qf_max),
            ) = SAC7.update_critic(
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                encoder_state,
                action_encoder_state,
                slice(data.observations),
                slice(data.actions),
                slice(data.next_observations),
                slice(data.rewards),
                slice(data.dones),
                key,
                min_qf_target,
                max_qf_target,
            )
            qf_state = SAC7.soft_update(tau, qf_state)
            encoder_state = SAC7.soft_update(tau, encoder_state)
            action_encoder_state = SAC7.soft_update(tau, action_encoder_state)

            min_qf_target += tau * (qf_min - min_qf_target)
            max_qf_target += tau * (qf_max - max_qf_target)

            # hack to be able to jit (n_updates % policy_delay == 0)
            if i in policy_delay_indices:
                (actor_state, qf_state, actor_loss_value, key, entropy) = cls.update_actor(
                    actor_state,
                    qf_state,
                    ent_coef_state,
                    encoder_state,
                    action_encoder_state,
                    slice(data.observations),
                    key,
                )
                ent_coef_state, _ = SAC7.update_temperature(target_entropy, ent_coef_state, entropy)

        return (
            qf_state,
            actor_state,
            ent_coef_state,
            encoder_state,
            action_encoder_state,
            key,
            (actor_loss_value, qf_loss_value, ent_coef_value),
            (min_qf_target, max_qf_target),
        )
