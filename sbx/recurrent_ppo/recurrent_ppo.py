import warnings
from functools import partial
from copy import deepcopy
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch as th
import gymnasium as gym
from flax.training.train_state import TrainState
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.vec_env import VecEnv

from sbx.common.on_policy_algorithm import OnPolicyAlgorithmJax
from sbx.common.recurrent import RecurrentRolloutBuffer, LSTMStates
from sbx.recurrent_ppo.policies import RecurrentPPOPolicy as PPOPolicy
from sbx.recurrent_ppo.policies import ScanLSTM

RPPOSelf = TypeVar("RPPOSelf", bound="RecurrentPPO")


class RecurrentPPO(OnPolicyAlgorithmJax):
    # TODO : Update documentation
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[PPOPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": PPOPolicy,
        # "CnnPolicy": ActorCriticCnnPolicy,
        # "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    policy: PPOPolicy  # type: ignore[assignment]

    def __init__(
        self,
        policy: Union[str, Type[PPOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            # Note: gSDE is not properly implemented,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # super()._setup_model()
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if not hasattr(self, "policy") or self.policy is None:  # type: ignore[has-type]
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )

            self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

            # TODO : what is ent_key ?
            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor
            self.vf = self.policy.vf

        # added the lstm states hidden_size (used to initialize the lstm states and the replay buffer)
        # at the moment just match the n_units of the policy
        # TODO : change this to enable more complex architectures (fix the n_lstm layers to 1 for now)
        self.hidden_state_size = self.policy.actor.n_units
        num_lstm_layers = 1
        lstm_state_buffer_shape = (self.n_steps, num_lstm_layers, self.n_envs, self.hidden_state_size)

        # use dummy lstm states to init the pi and the vf states 
        cell_hidden_lstm_states = ScanLSTM.initialize_carry(self.n_envs, self.hidden_state_size)
        # add them to the global LSTMStates
        init_lstm_states = LSTMStates(
            pi=cell_hidden_lstm_states,
            vf=cell_hidden_lstm_states,
        )
        # update the last lstm states (like in sb3 contrib)
        self._last_lstm_states = init_lstm_states 

        # Initialize the rollout buffer (it also encompasses the dones now as well as the lstm states)
        self.rollout_buffer = RecurrentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            lstm_state_buffer_shape=lstm_state_buffer_shape,
            device="cpu",
        )

        # Initialize schedules for policy/value clipping
        self.clip_range_schedule = get_schedule_fn(self.clip_range)
        # if self.clip_range_vf is not None:
        #     if isinstance(self.clip_range_vf, (float, int)):
        #         assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
        #
        #     self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"  # type: ignore[has-type]
        # Switch to eval mode (this affects batch norm / dropout)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise()

        callback.on_rollout_start()

        # copied that from sb3 contrib
        lstm_states = deepcopy(self._last_lstm_states)
        dones = self._last_episode_starts

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise()

            if not self.use_sde or isinstance(self.action_space, gym.spaces.Discrete):
                # Always sample new stochastic action
                self.policy.reset_noise()

            obs_tensor, _ = self.policy.prepare_obs(self._last_obs)  # type: ignore[has-type]
            # use the predict_all method with the lstm states and the dones
            actions, log_probs, values, lstm_states = self.policy.predict_all(obs_tensor, dones, lstm_states, self.policy.noise_key)

            actions = np.array(actions)
            log_probs = np.array(log_probs)
            values = np.array(values)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            
            # will be used to boostrap with the value function if need (need to to a critic pass)
            vf_lstm_states = lstm_states.vf
            lstm_in = (obs_tensor[np.newaxis, :], dones[np.newaxis, :])

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            # TODO : See how we can handle the lstm states here (bc we iterate on the dones)
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.prepare_obs(infos[idx]["terminal_observation"])[0]

                    # TODO Normally should only give the obs and dones for current idx
                    # TODO Should maybe pre-compute the lstm states and values before and then just iterate over the idx when needed
                    # TODO This is surely slowing everything for no reason (and seems false)
                    vf_lstm_states, values = self.vf.apply(
                        self.policy.vf_state.params,
                        vf_lstm_states,
                        lstm_in
                    )
                    terminal_value = values.flatten().item()
                    rewards[idx] += self.gamma * terminal_value

            # add the dones and the lstm states to the rollout buffer
            rollout_buffer.add(
                self._last_obs,  # type: ignore
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore
                th.as_tensor(values),
                th.as_tensor(log_probs),
                dones=dones,
                lstm_states=self._last_lstm_states,
            )

            # update the lstm states for the next iteration
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        # Get the last values when the rollout ends to compute the advantages
        vf_lstm_states = lstm_states.vf
        lstm_in = (self.policy.prepare_obs(new_obs)[0][np.newaxis, :], dones[np.newaxis, :])
        vf_lstm_states, values = self.vf.apply(
                        self.policy.vf_state.params,
                        vf_lstm_states,
                        lstm_in
                    )
        values = np.array(values).flatten()

        rollout_buffer.compute_returns_and_advantage(last_values=th.as_tensor(values), dones=dones)

        callback.on_rollout_end()

        return True

    @staticmethod
    @partial(jax.jit, static_argnames=["normalize_advantage"])
    def _one_update(
        actor_state: TrainState,
        vf_state: TrainState,
        lstm_states: LSTMStates,
        observations: np.ndarray,
        dones: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_prob: np.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
    ):
        # Normalize advantage
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # TODO : something weird here because the params argument isn't used and only actor_state.params instead
        def actor_loss(params):
            # TODO : see why I need to flatten dones here (otherwise error in the shapes given to the lstm)
            lstm_in = (observations[np.newaxis, :], dones.flatten()[np.newaxis, :])
            act_lstm_states, _ = lstm_states

            # TODO 
            # act_lstm_states, dist = actor_state.apply_fn(actor_state.params, act_lstm_states, lstm_in)
            act_lstm_states, dist = actor_state.apply_fn(params, act_lstm_states, lstm_in)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(log_prob - old_log_prob)
            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss

        pg_loss_value, pg_grads = jax.value_and_grad(actor_loss, has_aux=False)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=pg_grads)

        # TODO : same observation as above
        def critic_loss(params):            
            lstm_in = (observations[np.newaxis, :], dones.flatten()[np.newaxis, :])   
            _, vf_lstm_states = lstm_states
            # Value loss using the TD(gae_lambda) target
            # TODO 
            # vf_lstm_states, values = vf_state.apply_fn(vf_state.params, vf_lstm_states, lstm_in)
            vf_lstm_states, values = vf_state.apply_fn(params, vf_lstm_states, lstm_in)
            vf_values = values.flatten()
            return ((returns - vf_values) ** 2).mean()

        vf_loss_value, vf_grads = jax.value_and_grad(critic_loss, has_aux=False)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=vf_grads)

        # loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        return (actor_state, vf_state), (pg_loss_value, vf_loss_value)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range_schedule(self._current_progress_remaining)

        # train for n_epochs epochs
        for _ in range(self.n_epochs):
            # JIT only one update
            for rollout_data in self.rollout_buffer.get(self.batch_size):  # type: ignore[attr-defined]
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = rollout_data.actions.flatten().numpy().astype(np.int32)
                else:
                    actions = rollout_data.actions.numpy()

                # TODO : fix the values of the lstm states (at the moment they dot not follow the right temporal order I think)
                # TODO : also fix this mechanism where I need to reshape the lstm states here
                lstm_states_pi = (
                    rollout_data.lstm_states[0][0].numpy().reshape(self.batch_size, self.hidden_state_size),
                    rollout_data.lstm_states[0][1].numpy().reshape(self.batch_size, self.hidden_state_size)
                )

                lstm_states_vf = (
                    rollout_data.lstm_states[1][0].numpy().reshape(self.batch_size, self.hidden_state_size),
                    rollout_data.lstm_states[1][1].numpy().reshape(self.batch_size, self.hidden_state_size)
                )

                lstm_states = LSTMStates(
                    pi=lstm_states_pi,
                    vf=lstm_states_vf,
                )

                (self.policy.actor_state, self.policy.vf_state), (pg_loss, value_loss) = self._one_update(
                    actor_state=self.policy.actor_state,
                    vf_state=self.policy.vf_state,
                    observations=rollout_data.observations.numpy(),
                    actions=actions,
                    # added the dones here
                    dones=rollout_data.dones.numpy(),
                    advantages=rollout_data.advantages.numpy(),
                    returns=rollout_data.returns.numpy(),
                    old_log_prob=rollout_data.old_log_prob.numpy(),
                    # added the lstm states here
                    lstm_states=lstm_states,
                    clip_range=clip_range,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    normalize_advantage=self.normalize_advantage,
                )

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),  # type: ignore[attr-defined]
            self.rollout_buffer.returns.flatten(),  # type: ignore[attr-defined]
        )

        # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # TODO: use mean instead of one point
        self.logger.record("train/value_loss", value_loss.item())
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/pg_loss", pg_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: RPPOSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> RPPOSelf:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


if __name__ == "__main__":
    import gymnasium as gym
    from stable_baselines3.common.env_util import make_vec_env

    n_steps = 128
    batch_size = 32 
    train_steps = 10_000
    test_steps = 10
    n_envs = 4
    env_id = "CartPole-v1"

    # create vec env and train algo
    vec_env = make_vec_env(env_id, n_envs=n_envs)
    model = RecurrentPPO("MlpPolicy", vec_env, n_steps=n_steps, batch_size=batch_size, verbose=1)
    model.learn(total_timesteps=train_steps, progress_bar=True)

    vec_env = model.get_env()
    obs = vec_env.reset()
    lstm_states = None

    for _ in range(test_steps):
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

    vec_env.close()
