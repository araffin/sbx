import io
import pathlib
import warnings
from typing import Any

import jax
import numpy as np
import optax
from gymnasium import spaces
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer, NStepReplayBuffer, ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import DeserializationMode, GymEnv, Schedule
from stable_baselines3.common.utils import get_device


class OffPolicyAlgorithmJax(OffPolicyAlgorithm):
    qf_learning_rate: float

    def __init__(
        self,
        policy: type[BasePolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule,
        qf_learning_rate: float | None = None,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_kwargs: dict[str, Any] | None = None,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        device: str = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: int | None = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        stats_window_size: int = 100,
        param_resets: list[int] | None = None,
        supported_action_spaces: tuple[type[spaces.Space], ...] | None = None,
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
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            action_noise=action_noise,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            stats_window_size=stats_window_size,
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
        # Note: we do not allow separate schedule for it
        self.initial_qf_learning_rate = qf_learning_rate
        self.param_resets = param_resets
        self.reset_idx = 0

    def _maybe_reset_params(self) -> None:
        # Maybe reset the parameters
        if (
            self.param_resets
            and self.reset_idx < len(self.param_resets)
            and self.num_timesteps >= self.param_resets[self.reset_idx]
        ):
            # Note: we are not resetting the entropy coeff
            assert isinstance(self.qf_learning_rate, float)
            self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)  # type: ignore[operator]
            self.reset_idx += 1

    def _get_torch_save_params(self):
        return [], []

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        excluded.remove("policy")
        return excluded

    def _update_learning_rate(  # type: ignore[override]
        self,
        optimizers: list[optax.OptState] | optax.OptState,
        learning_rate: float,
        name: str = "learning_rate",
    ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers: An optimizer or a list of optimizers.
        :param learning_rate: The current learning rate to apply
        :param name: (Optional) A custom name for the lr (for instance qf_learning_rate)
        """
        # Log the current learning rate
        self.logger.record(f"train/{name}", learning_rate)

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            # Note: the optimizer must have been defined with inject_hyperparams
            optimizer.hyperparams["learning_rate"] = learning_rate

    def set_random_seed(self, seed: int | None) -> None:  # type: ignore[override]
        super().set_random_seed(seed)
        if seed is None:
            # Sample random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)

    def _setup_model(self) -> None:
        if self.replay_buffer_class is None:  # type: ignore[has-type]
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
                assert self.n_steps == 1, "N-step returns are not supported for Dict observation spaces yet."
            elif self.n_steps > 1:
                self.replay_buffer_class = NStepReplayBuffer
                # Add required arguments for computing n-step returns
                self.replay_buffer_kwargs.update({"n_steps": self.n_steps, "gamma": self.gamma})
            else:
                self.replay_buffer_class = ReplayBuffer

        self._setup_lr_schedule()
        # By default qf_learning_rate = pi_learning_rate
        self.qf_learning_rate = self.initial_qf_learning_rate or self.lr_schedule(1)
        self.set_random_seed(self.seed)
        # Make a local copy as we should not pickle
        # the environment when using HerReplayBuffer
        replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
        if issubclass(self.replay_buffer_class, HerReplayBuffer):  # type: ignore[arg-type]
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
            replay_buffer_kwargs["env"] = self.env

        self.replay_buffer = self.replay_buffer_class(  # type: ignore[misc]
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

    def load_replay_buffer(
        self,
        path: str | pathlib.Path | io.BufferedIOBase,
        truncate_last_traj: bool = True,
        deserialization_mode: DeserializationMode = DeserializationMode.LEGACY,
    ) -> None:
        """
        Load the replay buffer from a path.

        Uses ``deserialization_mode="legacy"`` (unrestricted pickle) by default because
        SBX uses JAX/Flax types not in the SB3 safe allowlist.

        :param path: Path to the replay buffer file.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If true, truncate the last trajectory to not include the full episode.
        :param deserialization_mode: How to handle deserialization.
            Default is ``"legacy"``. ``"safe"`` will fail for SBX buffers.
        """
        super().load_replay_buffer(path, truncate_last_traj, deserialization_mode=deserialization_mode)
        # Override replay buffer device to be always cpu for conversion to numpy
        assert self.replay_buffer is not None
        self.replay_buffer.device = get_device("cpu")

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        deserialization_mode: DeserializationMode = DeserializationMode.LEGACY,
        **kwargs,
    ):
        """
        Load the model from a zip-file.

        Uses ``deserialization_mode="legacy"`` (unrestricted pickle) by default because
        SBX models contain JAX/Flax types (TrainState, flax.linen modules) that are not
        in the SB3 safe deserialization allowlist.

        Warning: The legacy mode executes arbitrary Python code during deserialization.
        Only load SBX checkpoints from trusted sources.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace upon loading.
        :param print_system_info: Whether to print system info from the saved model
        :param force_reset: Force call to ``reset()`` before training
        :param deserialization_mode: How to handle deserialization.
            Default is ``"legacy"`` (unrestricted pickle). ``"safe"`` will fail
            because SBX uses JAX/Flax types not in the SB3 allowlist.
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if deserialization_mode == DeserializationMode.SAFE:
            warnings.warn(
                "Loading SBX model with deserialization_mode='safe'. "
                "SBX uses JAX/Flax types that are not in the SB3 safe allowlist. "
                "Loading will fail. Use deserialization_mode='legacy' (the default).",
                UserWarning,
            )
        return super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            deserialization_mode=deserialization_mode,
            **kwargs,
        )
