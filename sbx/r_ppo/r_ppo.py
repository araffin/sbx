import warnings
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from sbx.common.on_policy_algorithm import OnPolicyAlgorithmJax
# TODO : Fix this import 
from sbx.r_ppo.policies import RPPOPolicy as PPOPolicy

RPPOSelf = TypeVar("RPPOSelf", bound="RPPO")


class RPPO(OnPolicyAlgorithmJax):
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
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:  # type: ignore[has-type]
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )

            self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

            self.key, ent_key = jax.random.split(self.key, 2)

            self.actor = self.policy.actor
            self.vf = self.policy.vf

        # Initialize schedules for policy/value clipping
        self.clip_range_schedule = get_schedule_fn(self.clip_range)
        # if self.clip_range_vf is not None:
        #     if isinstance(self.clip_range_vf, (float, int)):
        #         assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
        #
        #     self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    # TODO : use lstm train state
    @staticmethod
    @partial(jax.jit, static_argnames=["normalize_advantage"])
    def _one_update(
        actor_state: TrainState,
        vf_state: TrainState,
        lstm_train_state: TrainState,
        observations: np.ndarray,
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

        # TODO : Maybe adding an lstm inside is easier for the gradients
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
            return total_policy_loss

        pg_loss_value, pg_grads = jax.value_and_grad(actor_loss, has_aux=False)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=pg_grads)

        def critic_loss(params):
            # Value loss using the TD(gae_lambda) target
            vf_values = vf_state.apply_fn(params, observations).flatten()
            return ((returns - vf_values) ** 2).mean()

        vf_loss_value, vf_grads = jax.value_and_grad(critic_loss, has_aux=False)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=vf_grads)

        # TODO ? What should be lstm loss ?? Atm just give as a loss the sum of losses for actor and critic
        lstm_grads = pg_grads + vf_grads
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
            # TODO : Fix the buffer here because we don't want to do permutations in it
            for rollout_data in self.rollout_buffer.get(self.batch_size):  # type: ignore[attr-defined]
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = rollout_data.actions.flatten().numpy().astype(np.int32)
                else:
                    actions = rollout_data.actions.numpy()

                (self.policy.actor_state, self.policy.vf_state), (pg_loss, value_loss) = self._one_update(
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
    from sbx import PPO

    # env = gym.make("CartPole-v1", render_mode="human")
    n_steps = 2048
    batch_size = 32
    train_steps = 5_000
    env = gym.make("CartPole-v1")

    model = PPO("MlpPolicy", env, n_steps=n_steps, batch_size=batch_size, verbose=1)
    vec_env = model.get_env()
    print("")
    print(f"{vec_env = }")
    obs = vec_env.reset()
    print(f"{obs = }")
    print(f"{obs.shape = }")


    model = RPPO("MlpPolicy", env, n_steps=n_steps, batch_size=batch_size, verbose=1)
    model.learn(total_timesteps=train_steps, progress_bar=True)

    vec_env = model.get_env()
    print(f"\n{vec_env = }")
    print("AA")
    obs = vec_env.reset()
    test_steps = 10
    for _ in range(test_steps):
        # vec_env.render()
        action, _states = model.predict(obs, deterministic=True)
        print(f"\n{action.shape = }")
        print(f"{obs.shape = }")
        print(f"{_states = }")
        obs, reward, done, info = vec_env.step(action)

    vec_env.close()