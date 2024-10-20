import functools

from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import BaseJaxPolicy, Flatten
from sbx.common.recurrent import LSTMStates

tfd = tfp.distributions


# Added a ScanLSTM Module that automatically handles the reset of LSTM states
# inspired from the ScanRNN in purejaxrl : https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo_rnn.py
class ScanLSTM(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast='params',
        in_axes=0,
        out_axes=0,
        split_rngs={'params': False}
    )
    @nn.compact
    def __call__(self, lstm_states, inputs_and_resets):
        # pass the pi and vf lstm states, as well as the obs and the resets
        input, resets = inputs_and_resets
        hidden_state, cell_state = lstm_states

        # create new lstm states to replace the old ones if reset is True
        reset_lstm_states = self.initialize_carry(hidden_state.shape[0], hidden_state.shape[1])

        # handle the reset of the hidden lstm states
        hidden_state = jnp.where(
            resets[:, np.newaxis],
            reset_lstm_states[0],
            hidden_state
        )
        # handle the reset of the cell lstm states
        cell_state = jnp.where(
            resets[:, np.newaxis],
            reset_lstm_states[1],
            cell_state
        )

        lstm_states = (hidden_state, cell_state)
        hidden_size = lstm_states[0].shape[-1]
        
        new_lstm_states, output = nn.LSTMCell(features=hidden_size)(lstm_states, input)
        return new_lstm_states, output
    
    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Returns a tuple of lstm states (hidden and cell states)
        return nn.LSTMCell(features=hidden_size).initialize_carry(
            rng=jax.random.PRNGKey(0), input_shape=(batch_size, hidden_size)
        )

# Add ScanLSTM as first element of the Critic architecture
class Critic(nn.Module):
    n_units: int = 256
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh

    # return hidden state + val
    @nn.compact
    def __call__(self, lstm_states, obs_dones) -> jnp.ndarray:
        lstm_states, out = ScanLSTM()(lstm_states, obs_dones)
        x = nn.Dense(self.n_units)(out)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return lstm_states, x

# Add ScanLSTM as first element of the Actor architecture
class Actor(nn.Module):
    action_dim: int
    n_units: int = 256
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    # For Discrete, MultiDiscrete and MultiBinary actions
    num_discrete_choices: Optional[Union[int, Sequence[int]]] = None
    # For MultiDiscrete
    max_num_choices: int = 0
    split_indices: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_std(self) -> jnp.ndarray:
        # Make it work with gSDE
        return jnp.array(0.0)

    def __post_init__(self) -> None:
        # For MultiDiscrete
        if isinstance(self.num_discrete_choices, np.ndarray):
            self.max_num_choices = max(self.num_discrete_choices)
            # np.cumsum(...) gives the correct indices at which to split the flatten logits
            self.split_indices = np.cumsum(self.num_discrete_choices[:-1])
        super().__post_init__()

    # return hidden state + action dist
    @nn.compact
    def __call__(self, hidden, obs_dones) -> tfd.Distribution:  # type: ignore[name-defined]
        hidden, out = ScanLSTM()(hidden, obs_dones)
        out = jnp.squeeze(out, axis=0)

        x = nn.Dense(self.n_units)(out)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        action_logits = nn.Dense(self.action_dim)(x)
        if self.num_discrete_choices is None:
            # Continuous actions
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))
        elif isinstance(self.num_discrete_choices, int):
            dist = tfd.Categorical(logits=action_logits)
        else:
            # Split action_logits = (batch_size, total_choices=sum(self.num_discrete_choices))
            action_logits = jnp.split(action_logits, self.split_indices, axis=1)
            # Pad to the maximum number of choices (required by tfp.distributions.Categorical).
            # Pad by -inf, so that the probability of these invalid actions is 0.
            logits_padded = jnp.stack(
                [
                    jnp.pad(
                        logit,
                        # logit is of shape (batch_size, n)
                        # only pad after dim=1, to max_num_choices - n
                        # pad_width=((before_dim_0, after_0), (before_dim_1, after_1))
                        pad_width=((0, 0), (0, self.max_num_choices - logit.shape[1])),
                        constant_values=-np.inf,
                    )
                    for logit in action_logits
                ],
                axis=1,
            )
            dist = tfp.distributions.Independent(
                tfp.distributions.Categorical(logits=logits_padded), reinterpreted_batch_ndims=1
            )
        return hidden, dist


# TODO Later : at the moment custom net_architectures are not supported for the LSTM
class RecurrentPPOPolicy(BaseJaxPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        ortho_init: bool = False,
        log_std_init: float = 0.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
    ):
        if optimizer_kwargs is None:
            # Small values to avoid NaN in Adam optimizer
            optimizer_kwargs = {}
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.n_units = net_arch[0]
            else:
                assert isinstance(net_arch, dict)
                self.n_units = net_arch["pi"][0]
        else:
            self.n_units = 64
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.Array, lr_schedule: Schedule, max_grad_norm: float) -> jax.Array:
        key, actor_key, vf_key = jax.random.split(key, 3)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs = {
                "action_dim": int(np.prod(self.action_space.shape)),
            }
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": int(self.action_space.n),
                "num_discrete_choices": int(self.action_space.n),
            }
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            assert self.action_space.nvec.ndim == 1, (
                f"Only one-dimensional MultiDiscrete action spaces are supported, "
                f"but found MultiDiscrete({(self.action_space.nvec).tolist()})."
            )
            actor_kwargs = {
                "action_dim": int(np.sum(self.action_space.nvec)),
                "num_discrete_choices": self.action_space.nvec,  # type: ignore[dict-item]
            }
        elif isinstance(self.action_space, spaces.MultiBinary):
            assert isinstance(self.action_space.n, int), (
                f"Multi-dimensional MultiBinary({self.action_space.n}) action space is not supported. "
                "You can flatten it instead."
            )
            # Handle binary action spaces as discrete action spaces with two choices.
            actor_kwargs = {
                "action_dim": 2 * self.action_space.n,
                "num_discrete_choices": 2 * np.ones(self.action_space.n, dtype=int),
            }
        else:
            raise NotImplementedError(f"{self.action_space}")
        

        self.actor = Actor(
            n_units=self.n_units,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            **actor_kwargs,  # type: ignore[arg-type]
        )

        # Initialize a dummy input for the LSTM layer (obs, dones)
        init_obs = jnp.array([self.observation_space.sample()])
        init_dones = jnp.zeros((init_obs.shape[0],))
        # at the moment use this trick of adding a dimension to obs and dones to pass them to the LSTM
        init_x = (init_obs[np.newaxis, :], init_dones[np.newaxis, :])

        # hardcode the number of envs to 1 for the initialization of the lstm states
        n_envs = 1
        hidden_size = self.n_units 
        init_lstm_states = ScanLSTM.initialize_carry(n_envs, hidden_size)

        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        # pass the init lstm states as argument to the actor train state
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, init_lstm_states, init_x),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.vf = Critic(n_units=self.n_units, activation_fn=self.activation_fn)

        # pass the init lstm states as argument to the critic train state
        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init({"params": vf_key}, init_lstm_states, init_x),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, lstm_states, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    # TODO : Add the lstm state to the _predict_method (Might also need to return them)
    # Like in this recurrent ppo ex in sb3 contrib : https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html 
    def _predict(self, observation: np.ndarray, lstm_states, deterministic: bool = False) -> np.ndarray:  # type: ignore[override]
        if deterministic:
            # TODO : pass the lstm state here (see how to do it cleanly because uses a function from parent class)
            return BaseJaxPolicy.select_action(self.actor_state, observation)
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
            # TODO : also include lstm state here
        return BaseJaxPolicy.sample_action(self.actor_state, observation, self.noise_key)

    # Added the lstm states to the predict_all method (maybe also the dones but I don't remember)
    def predict_all(self, observation: np.ndarray, done, lstm_states, key: jax.Array) -> np.ndarray:
        return self._predict_all(self.actor_state, self.vf_state, observation, done, lstm_states, key)

    @staticmethod
    @jax.jit
    def _predict_all(actor_state, vf_state, observations, dones, lstm_states, key):
        # separate the lstm states for the actor and the critic, and prepare the input for the lstm
        act_lstm_states, vf_lstm_states = lstm_states
        lstm_in = (observations[np.newaxis, :], dones[np.newaxis, :])

        # pass the actor lstm states and the input to the actor
        act_lstm_states, dist = actor_state.apply_fn(actor_state.params, act_lstm_states, lstm_in)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)

        # pass the critic lstm states and the input to the critic
        vf_lstm_states, values = vf_state.apply_fn(vf_state.params, vf_lstm_states, lstm_in)
        values = values.flatten()

        # add the actor and critic lstm states to the lstm states tuple
        lstm_states = LSTMStates(
            pi=act_lstm_states,
            vf=vf_lstm_states
        )

        return actions, log_probs, values, lstm_states
