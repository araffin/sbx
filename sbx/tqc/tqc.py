import copy
from functools import partial
from typing import Dict, NamedTuple, Optional, Tuple, Union

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import is_vectorized_observation

from sbx.tqc.policies import Actor, Critic


class ReplayBufferSamplesNp(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    rewards: np.ndarray


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param("log_ent_coef", init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)))
        return jnp.exp(log_ent_coef)


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict


@partial(jax.jit, static_argnames="actor")
def sample_action(actor, actor_state, obervations, key):
    dist = actor.apply(actor_state.params, obervations)
    action = dist.sample(seed=key)
    return action


@partial(jax.jit, static_argnames="actor")
def select_action(actor, actor_state, obervations):
    return actor.apply(actor_state.params, obervations).mode()


class TQC(OffPolicyAlgorithm):

    policy_aliases: Dict[str, Optional[nn.Module]] = {
        "MlpPolicy": None,
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
        gradient_steps: int = 1,
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
            gradient_steps=gradient_steps,
            verbose=verbose,
            seed=seed,
        )
        # self.agent = agent
        # Will be updated later
        self.key = jax.random.PRNGKey(0)
        self.dropout_rate = 0.0
        self.layer_norm = False
        self.n_units = 256
        self.top_quantiles_to_drop_per_net = 2
        self.squash_output = True

        if _init_setup_model:
            self._setup_model()

    def _get_torch_save_params(self):
        state_dicts = []
        return state_dicts, []

    def set_random_seed(self, seed: int) -> None:
        super().set_random_seed(seed)
        if seed is None:
            # Sample random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.replay_buffer_class = ReplayBuffer
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs,
        )
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        # self.policy = self.policy_class(  # pytype:disable=not-instantiable
        #     self.observation_space,
        #     self.action_space,
        #     self.lr_schedule,
        #     **self.policy_kwargs,  # pytype:disable=not-instantiable
        # )
        # self.policy = self.policy.to(self.device)
        self.key, actor_key, qf1_key, qf2_key = jax.random.split(self.key, 4)
        self.key, dropout_key1, dropout_key2, ent_key = jax.random.split(self.key, 4)

        obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.actor = Actor(
            action_dim=np.prod(self.action_space.shape),
            n_units=self.n_units,
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.adam(learning_rate=self.learning_rate),
        )

        ent_coef_init = 1.0
        self.ent_coef = EntropyCoef(ent_coef_init)
        self.ent_coef_state = TrainState.create(
            apply_fn=self.ent_coef.apply,
            params=self.ent_coef.init(ent_key)["params"],
            tx=optax.adam(
                learning_rate=self.learning_rate,
            ),
        )

        # Sort and drop top k quantiles to control overestimation.
        n_quantiles = 25
        n_critics = 2
        quantiles_total = n_quantiles * n_critics
        top_quantiles_to_drop_per_net = self.top_quantiles_to_drop_per_net
        self.n_target_quantiles = quantiles_total - top_quantiles_to_drop_per_net * n_critics

        # automatically set target entropy if needed
        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

        self.qf = Critic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            n_units=self.n_units,
            n_quantiles=n_quantiles,
        )

        self.qf1_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf1_key, "dropout": dropout_key1},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf1_key, "dropout": dropout_key1},
                obs,
                action,
            ),
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.qf2_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf2_key, "dropout": dropout_key2},
                obs,
                action,
            ),
            target_params=self.qf.init(
                {"params": qf2_key, "dropout": dropout_key2},
                obs,
                action,
            ),
            tx=optax.adam(learning_rate=self.learning_rate),
        )
        self.actor.apply = jax.jit(self.actor.apply)
        self.qf.apply = jax.jit(self.qf.apply, static_argnames=("dropout_rate", "use_layer_norm"))

        obs = self.env.reset()
        real_next_obs, rewards, dones, infos = self.env.step(action)
        self.replay_buffer.add(obs, real_next_obs, action, rewards, dones, infos)
        self.train()
        print(self.predict(obs, deterministic=False)[0])
        print(self.predict(obs, deterministic=False)[0])
        print(self.predict(obs, deterministic=True)[0])
        print(self.predict(obs, deterministic=True)[0])
        self.train()
        print(self.predict(obs, deterministic=True)[0])

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
        pass
        # return super().learn(
        #     total_timesteps=total_timesteps,
        #     callback=callback,
        #     log_interval=log_interval,
        #     eval_env=eval_env,
        #     eval_freq=eval_freq,
        #     n_eval_episodes=n_eval_episodes,
        #     tb_log_name=tb_log_name,
        #     eval_log_path=eval_log_path,
        #     reset_num_timesteps=reset_num_timesteps,
        # )

    def _predict(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if deterministic:
            return select_action(self.actor, self.actor_state, observation)
        self.key, noise_key = jax.random.split(self.key, 2)
        return sample_action(self.actor, self.actor_state, observation, noise_key)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1,) + self.action_space.shape)

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                # actions = self.unscale_action(actions)
                # TODO: move that to policy
                pass
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def prepare_obs(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            observation = copy.deepcopy(observation)
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                if is_image_space(obs_space):
                    obs_ = maybe_transpose(obs, obs_space)
                else:
                    obs_ = np.array(obs)
                vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
                # Add batch dimension if needed
                observation[key] = obs_.reshape((-1,) + self.observation_space[key].shape)

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(observation, dict):
            # Dict obs need to be handled separately
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1,) + self.observation_space.shape)

        return observation, vectorized_env

    def train(self):
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(self.batch_size * self.gradient_steps)
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
            self.qf1_state,
            self.qf2_state,
            self.actor_state,
            self.ent_coef_state,
            self.key,
            (qf1_loss_value, qf2_loss_value, actor_loss_value),
        ) = self._train(
            self.actor,
            self.qf,
            self.ent_coef,
            self.gamma,
            self.tau,
            self.target_entropy,
            self.gradient_steps,
            self.n_target_quantiles,
            data,
            n_updates,
            self.qf1_state,
            self.qf2_state,
            self.actor_state,
            self.ent_coef_state,
            self.key,
        )

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
        # TODO Maybe pre-generate a lot of random keys
        # also check https://jax.readthedocs.io/en/latest/jax.random.html
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
            (qf1_state, qf2_state, ent_coef_state),
            (qf1_loss_value, qf2_loss_value),
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

            def slice(x):
                assert x.shape[0] % gradient_steps == 0
                # batch_size = batch_size
                batch_size = x.shape[0] // gradient_steps
                return x[batch_size * i : batch_size * (i + 1)]

            ((qf1_state, qf2_state, ent_coef_state), (qf1_loss_value, qf2_loss_value), key,) = TQC.update_critic(
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
            (qf1_loss_value, qf2_loss_value, actor_loss_value),
        )


def main():

    # eval_envs = make_vec_env(args.env_id, n_envs=args.n_eval_envs, seed=args.seed)

    # agent = Agent(actor_state)
    # agent.select_action = select_action

    model = TQC("MlpPolicy", "Pendulum-v1", seed=0)

    # model = TQC.load("test_save")
    # model.agent.select_action = select_action
    # print(evaluate_policy(model.agent, eval_envs, n_eval_episodes=args.n_eval_episodes))
    #
    exit()
    # start_time = time.time()
    # n_updates = 0
    # # for global_step in range(args.total_timesteps):
    # for global_step in tqdm(range(args.total_timesteps)):
    #     # ALGO LOGIC: put action logic here
    #     if global_step < args.learning_starts:
    #         actions = np.array([envs.action_space.sample() for _ in range(envs.num_envs)])
    #     else:
    #         key, exploration_key = jax.random.split(key, 2)
    #         actions = np.array(sample_action(actor_state, obs, exploration_key))
    #
    #     # actions = np.clip(actions, -1.0, 1.0)
    #     # TRY NOT TO MODIFY: execute the game and log data.
    #     next_obs, rewards, dones, infos = envs.step(actions)
    #
    #     # TRY NOT TO MODIFY: save data to replay buffer; handle `terminal_observation`
    #     real_next_obs = next_obs.copy()
    #     for idx, done in enumerate(dones):
    #         if done:
    #             real_next_obs[idx] = infos[idx]["terminal_observation"]
    #         # Timeout handling done inside the replay buffer
    #         # if infos[idx].get("TimeLimit.truncated", False) == True:
    #         #     real_dones[idx] = False
    #
    #     rb.add(obs, real_next_obs, actions, rewards, dones, infos)
    #
    #     # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    #     obs = next_obs
    #
    #     # ALGO LOGIC: training.
    #     if global_step > args.learning_starts:
    #         # Sample all at once for efficiency (so we can jit the for loop)
    #         data = rb.sample(args.batch_size * args.gradient_steps)
    #         # Convert to numpy
    #         data = ReplayBufferSamplesNp(
    #             data.observations.numpy(),
    #             data.actions.numpy(),
    #             data.next_observations.numpy(),
    #             data.dones.numpy().flatten(),
    #             data.rewards.numpy().flatten(),
    #         )
    #
    #         (
    #             n_updates,
    #             qf1_state,
    #             qf2_state,
    #             actor_state,
    #             ent_coef_state,
    #             key,
    #             (qf1_loss_value, qf2_loss_value, actor_loss_value),
    #         ) = train(
    #             data,
    #             n_updates,
    #             qf1_state,
    #             qf2_state,
    #             actor_state,
    #             ent_coef_state,
    #             key,
    #         )
    #
    #         fps = int(global_step / (time.time() - start_time))
    #         if args.eval_freq > 0 and global_step % args.eval_freq == 0:
    #             agent.actor_state = actor_state
    #             mean_reward, std_reward = evaluate_policy(agent, eval_envs, n_eval_episodes=args.n_eval_episodes)
    #             print(f"global_step={global_step}, mean_eval_reward={mean_reward:.2f} +/- {std_reward:.2f} - {fps} fps")
    #             # writer.add_scalar("charts/mean_eval_reward", mean_reward, global_step)
    #             # writer.add_scalar("charts/std_eval_reward", std_reward, global_step)
    #
    #         # if global_step % 100 == 0:
    #         #     ent_coef_value = ent_coef.apply({"params": ent_coef_state.params})
    #         #     writer.add_scalar("losses/ent_coef_value", ent_coef_value.item(), global_step)
    #         #     writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
    #         #     writer.add_scalar("losses/qf2_loss", qf2_loss_value.item(), global_step)
    #         #     # writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
    #         #     # writer.add_scalar("losses/qf2_values", qf2_a_values.item(), global_step)
    #         #     writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
    #         #     if args.verbose >= 2:
    #         #         print("FPS:", fps)
    #         #     writer.add_scalar(
    #         #         "charts/SPS",
    #         #         int(global_step / (time.time() - start_time)),
    #         #         global_step,
    #         #     )
    #
    # # envs.close()
    # # writer.close()
    # model.save("test_save")


if __name__ == "__main__":

    try:
        main()
    except KeyboardInterrupt:
        pass
