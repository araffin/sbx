from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import jax
import numpy as np
from gym import spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule


class OffPolicyAlgorithmJax(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        qf_learning_rate: Optional[float] = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: str = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):
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
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            action_noise=action_noise,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
            support_multi_env=support_multi_env,
        )
        # Will be updated later
        self.key = jax.random.PRNGKey(0)
        # Note: we do not allow schedule for it
        self.qf_learning_rate = qf_learning_rate

    def _get_torch_save_params(self):
        return [], []

    def _excluded_save_params(self) -> List[str]:
        excluded = super()._excluded_save_params()
        excluded.remove("policy")
        return excluded

    def set_random_seed(self, seed: int) -> None:
        super().set_random_seed(seed)
        if seed is None:
            # Sample random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)

    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        self._setup_lr_schedule()
        # By default qf_learning_rate = pi_learning_rate
        self.qf_learning_rate = self.qf_learning_rate or self.lr_schedule(1)
        self.set_random_seed(self.seed)
        # Make a local copy as we should not pickle
        # the environment when using HerReplayBuffer
        replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        if issubclass(self.replay_buffer_class, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
            replay_buffer_kwargs["env"] = self.env
        self.replay_buffer = self.replay_buffer_class(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            device="cpu",  # force cpu device to easy torch -> numpy conversion
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **replay_buffer_kwargs,
        )
        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
