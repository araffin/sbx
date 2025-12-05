import warnings
from functools import partial
from typing import Any, ClassVar, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance

from sbx.common.on_policy_algorithm import OnPolicyAlgorithmJax
from sbx.common.utils import KLAdaptiveLR, copy_naturecnn_params
from sbx.ppo.policies import CnnPolicy, PPOPolicy

PPOSelf = TypeVar("PPOSelf", bound="PPO")


class PPO(OnPolicyAlgorithmJax):
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
    :param target_kl: Update the learning rate based on a desired KL divergence (see https://arxiv.org/abs/1707.02286).
        Note: this will overwrite any lr schedule.
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

    policy_aliases: ClassVar[dict[str, type[PPOPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": PPOPolicy,
        "CnnPolicy": CnnPolicy,
        # "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    policy: PPOPolicy  # type: ignore[assignment]
    adaptive_lr: KLAdaptiveLR

    def __init__(
        self,
        policy: str | type[PPOPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: None | float | Schedule = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: float | None = None,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
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
        # If set will trigger adaptive lr
        self.target_kl = target_kl
        if target_kl is not None and self.verbose > 0:
            print(f"Using adaptive learning rate with {target_kl=}, any other lr schedule will be skipped.")

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if self.target_kl is not None:
            self.adaptive_lr = KLAdaptiveLR(self.target_kl, self.lr_schedule(1.0))

        if not hasattr(self, "policy") or self.policy is None:  # type: ignore[has-type]
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )

            self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

            self.actor = self.policy.actor  # type: ignore[assignment]
            self.vf = self.policy.vf  # type: ignore[assignment]

        # Initialize schedules for policy/value clipping
        self.clip_range_schedule = FloatSchedule(self.clip_range)
        # if self.clip_range_vf is not None:
        #     if isinstance(self.clip_range_vf, (float, int)):
        #         assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
        #
        #     self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    @staticmethod
    @partial(jax.jit, static_argnames=["normalize_advantage", "share_features_extractor"])
    def _one_update(
        actor_state: TrainState,
        vf_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_log_prob: np.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
        share_features_extractor: bool = False,
    ):
        # Normalize advantage
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_loss(params):
            dist = actor_state.apply_fn(params, observations)
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
            return total_policy_loss, (ratio, policy_loss, entropy_loss)

        (pg_loss_value, (ratio, policy_loss, entropy_loss)), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params
        )
        actor_state = actor_state.apply_gradients(grads=grads)

        if share_features_extractor:
            # Hack: selective copy to share features extractor when using CNN
            vf_state = copy_naturecnn_params(actor_state, vf_state)

        def critic_loss(params):
            # Value loss using the TD(gae_lambda) target
            vf_values = vf_state.apply_fn(params, observations).flatten()
            return vf_coef * ((returns - vf_values) ** 2).mean()

        vf_loss_value, grads = jax.value_and_grad(critic_loss, has_aux=False)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)

        if share_features_extractor:
            actor_state = copy_naturecnn_params(vf_state, actor_state)

        # loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        return (actor_state, vf_state), (pg_loss_value, policy_loss, entropy_loss, vf_loss_value, ratio)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        if self.target_kl is None:
            self._update_learning_rate(
                [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                learning_rate=self.lr_schedule(self._current_progress_remaining),
            )
        # Compute current clip range
        clip_range = self.clip_range_schedule(self._current_progress_remaining)
        n_updates = 0
        mean_clip_fraction = 0.0
        mean_kl_div = 0.0

        # train for n_epochs epochs
        for _ in range(self.n_epochs):
            # JIT only one update
            for rollout_data in self.rollout_buffer.get(self.batch_size):  # type: ignore[attr-defined]
                n_updates += 1
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = rollout_data.actions.flatten().numpy().astype(np.int32)
                else:
                    actions = rollout_data.actions.numpy()

                (self.policy.actor_state, self.policy.vf_state), (pg_loss, policy_loss, entropy_loss, value_loss, ratio) = (
                    self._one_update(
                        actor_state=self.policy.actor_state,
                        vf_state=self.policy.vf_state,
                        observations=rollout_data.observations.numpy(),
                        actions=actions,
                        advantages=rollout_data.advantages.numpy(),
                        returns=rollout_data.returns.numpy(),
                        old_log_prob=rollout_data.old_log_prob.numpy(),
                        clip_range=clip_range,
                        ent_coef=self.ent_coef,
                        vf_coef=self.vf_coef,
                        normalize_advantage=self.normalize_advantage,
                        # Sharing the CNN between actor and critic has a great impact on performance
                        # for Atari games
                        share_features_extractor=isinstance(self.policy, CnnPolicy),
                    )
                )

                # Calculate approximate form of reverse KL Divergence for adaptive lr
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                eps = 1e-7  # Avoid NaN due to numerical instabilities
                approx_kl_div = jnp.mean((ratio - 1.0 + eps) - jnp.log(ratio + eps)).item()
                clip_fraction = jnp.mean(jnp.abs(ratio - 1) > clip_range).item()
                # Compute average
                mean_clip_fraction += (clip_fraction - mean_clip_fraction) / n_updates
                mean_kl_div += (approx_kl_div - mean_kl_div) / n_updates

                # Adaptive lr schedule, see https://arxiv.org/abs/1707.02286
                if self.target_kl is not None:
                    self.adaptive_lr.update(approx_kl_div)

                    self._update_learning_rate(
                        [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                        learning_rate=self.adaptive_lr.current_adaptive_lr,
                    )
        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),  # type: ignore[attr-defined]
            self.rollout_buffer.returns.flatten(),  # type: ignore[attr-defined]
        )

        # Logs
        # TODO: use mean instead of one point
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/approx_kl", mean_kl_div)
        self.logger.record("train/clip_fraction", mean_clip_fraction)
        self.logger.record("train/pg_loss", pg_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        try:
            log_std = self.policy.actor_state.params["params"]["log_std"]
            self.logger.record("train/std", np.exp(log_std).mean().item())
        except KeyError:
            pass
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: PPOSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> PPOSelf:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
