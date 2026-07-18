"""Safe deserialization registration for SBX types.

This module registers all SBX types (policies, distributions,
buffers, custom layers, train states, named tuples) with the SB3 safe deserialization
allowlist, so that SBX saved models can be loaded with
``deserialization_mode="safe"`` (the default since SB3 2.10).

The registration is performed automatically when sbx is imported.
"""

from __future__ import annotations

import torch as th
from stable_baselines3.common.safe_globals import add_safe_globals, get_safe_globals


def _register() -> None:
    """Register all sbx types with the safe deserialization allowlist."""
    # Import JAX/Flax/Optax types needed for safe deserialization
    import inspect

    import cloudpickle.cloudpickle
    import flax.ids
    import flax.linen.module
    import flax.training.train_state
    import jax._src.array
    import jax._src.custom_derivatives
    import jax._src.pjit
    import jax._src.tree_util
    import jax._src.named_sharding
    import jaxlib._jax
    import jaxlib._jax.pytree
    import optax

    from sbx.common.distributions import TanhTransformedDistribution
    from sbx.common.jax_layers import BatchRenorm, NatureCNN, SimbaResidualBlock
    from sbx.common.policies import (
        BaseJaxPolicy,
        ContinuousCritic,
        Flatten,
        SimbaContinuousCritic,
        SimbaSquashedGaussianActor,
        SimbaVectorCritic,
        SquashedGaussianActor,
        VectorCritic,
    )
    from sbx.common.type_aliases import (
        BatchNormTrainState,
        ReplayBufferSamplesNp,
        RLTrainState,
    )

    # Import SBX utility classes
    from sbx.common.utils import KLAdaptiveLR

    # Import CrossQ entropy coef classes
    from sbx.crossq.crossq import (
        ConstantEntropyCoef as CrossQConstantEntropyCoef,
    )
    from sbx.crossq.crossq import EntropyCoef as CrossQEntropyCoef
    from sbx.crossq.policies import (
        Actor as CrossQActor,
    )
    from sbx.crossq.policies import (
        Critic as CrossQCritic,
    )
    from sbx.crossq.policies import (
        CrossQPolicy,
        SimbaActor,
        SimbaCritic,
        SimbaCrossQPolicy,
    )
    from sbx.crossq.policies import (
        SimbaVectorCritic as CrossQSimbaVectorCritic,
    )
    from sbx.crossq.policies import (
        VectorCritic as CrossQVectorCritic,
    )
    from sbx.dqn.policies import (
        CNNPolicy as DQNCnnPolicy,
    )
    from sbx.dqn.policies import (
        CnnQNetwork,
        DQNPolicy,
        QNetwork,
    )
    from sbx.ppo.policies import (
        Actor as PPOActor,
    )
    from sbx.ppo.policies import (
        CnnPolicy as PPOCnnPolicy,
    )
    from sbx.ppo.policies import (
        Critic as PPOCritic,
    )
    from sbx.ppo.policies import (
        PPOPolicy,
    )
    from sbx.sac.policies import SACPolicy, SimbaSACPolicy

    # Import SAC entropy coef classes
    from sbx.sac.sac import ConstantEntropyCoef as SACConstantEntropyCoef
    from sbx.sac.sac import EntropyCoef as SACEntropyCoef
    from sbx.td3.policies import Actor as TD3Actor
    from sbx.td3.policies import TD3Policy
    from sbx.tqc.policies import SimbaTQCPolicy, TQCPolicy
    from sbx.tqc.tqc import ConstantEntropyCoef as TQCConstantEntropyCoef
    from sbx.tqc.tqc import EntropyCoef as TQCEntropyCoef

    sbx_types: list[type | tuple[type, str]] = [
        # Common distributions
        TanhTransformedDistribution,
        # Common jax layers
        BatchRenorm,
        NatureCNN,
        SimbaResidualBlock,
        # Common policies
        BaseJaxPolicy,
        Flatten,
        ContinuousCritic,
        SimbaContinuousCritic,
        VectorCritic,
        SimbaVectorCritic,
        SquashedGaussianActor,
        SimbaSquashedGaussianActor,
        # Common type aliases
        RLTrainState,
        BatchNormTrainState,
        ReplayBufferSamplesNp,
        # SAC policies
        SACPolicy,
        SimbaSACPolicy,
        # PPO policies
        PPOPolicy,
        PPOCnnPolicy,
        PPOActor,
        PPOCritic,
        # TD3 policies
        TD3Policy,
        TD3Actor,
        # DQN policies
        DQNPolicy,
        DQNCnnPolicy,
        QNetwork,
        CnnQNetwork,
        # CrossQ policies
        CrossQPolicy,
        SimbaCrossQPolicy,
        CrossQActor,
        CrossQCritic,
        CrossQVectorCritic,
        CrossQSimbaVectorCritic,
        SimbaActor,
        SimbaCritic,
        # TQC policies
        TQCPolicy,
        SimbaTQCPolicy,
        # TQC entropy coef
        TQCEntropyCoef,
        TQCConstantEntropyCoef,
        # SAC entropy coef
        SACEntropyCoef,
        SACConstantEntropyCoef,
        # CrossQ entropy coef
        CrossQEntropyCoef,
        CrossQConstantEntropyCoef,
        # Flax/Optax/JAX types needed for deserialization
        flax.ids.FlaxId,
        flax.linen.module._ModuleInternalState,
        flax.linen.module.Module.apply,  # type: ignore[list-item]
        flax.linen.module.SetupState,
        flax.training.train_state.TrainState,
        inspect.Signature,
        jax._src.array._reconstruct_array,  # type: ignore[list-item]
        jax._src.custom_derivatives.custom_jvp,
        jax._src.pjit.PjitInfo,  # type: ignore[list-item]
        jax._src.pjit._python_pjit_helper,  # type: ignore[list-item]
        jaxlib._jax.PjitFunction,
        # JAX PyTree types (needed for JAX array/structure serialization)
        jax._src.tree_util._RegistryEntry,  # type: ignore[list-item]
        jaxlib._jax.pytree.DictKey,  # type: ignore[list-item]
        jaxlib._jax.pytree.FlattenedIndexKey,  # type: ignore[list-item]
        jaxlib._jax.pytree.GetAttrKey,  # type: ignore[list-item]
        jaxlib._jax.pytree.PyTreeDef,  # type: ignore[list-item]
        jaxlib._jax.pytree.PyTreeRegistry,  # type: ignore[list-item]
        jaxlib._jax.pytree.SequenceKey,  # type: ignore[list-item]
        # Optax base classes and states
        optax._src.base.EmptyState,  # type: ignore[list-item]
        optax._src.base.GradientTransformation,  # type: ignore[list-item]
        optax._src.base.GradientTransformationExtraArgs,  # type: ignore[list-item]
        optax._src.base.init_empty_state,  # type: ignore[list-item]
        optax._src.transform.ScaleByAdamState,  # type: ignore[list-item]
        # Inspect types (for function signatures)
        inspect.Parameter,
        inspect._ParameterKind,
        inspect._empty,
        # Cloudpickle helper for importing modules (needed for deserializing functions)
        cloudpickle.cloudpickle.subimport,  # type: ignore[list-item]
        # Optax optimizers (functions, not types, but needed for pickle deserialization)
        optax.adam,  # type: ignore[list-item]
        optax.adamw,  # type: ignore[list-item]
        optax.sgd,  # type: ignore[list-item]
        optax.rmsprop,  # type: ignore[list-item]
        optax.adagrad,  # type: ignore[list-item]
        optax.adadelta,  # type: ignore[list-item]
        optax.radam,  # type: ignore[list-item]
        optax.lamb,  # type: ignore[list-item]
        optax.lars,  # type: ignore[list-item]
        optax.sm3,  # type: ignore[list-item]
        optax.yogi,  # type: ignore[list-item]
        optax.adan,  # type: ignore[list-item]
        # SBX utility classes
        KLAdaptiveLR,
    ]

    # Add all JAX pjit functions to the allowlist (needed for JAX jit/pjit serialization)
    for _name in (
        "_get_fastpath_data",
        "_need_to_rebuild_with_fdo",
        "_cpp_pjit",
        "_create_pjit_jaxpr",
        "_infer_params",
        "_pjit_call_impl",
        "_pjit_lower",
    ):
        if hasattr(jax._src.pjit, _name):
            sbx_types.append(getattr(jax._src.pjit, _name))  # type: ignore[list-item]

    add_safe_globals(sbx_types)
    th.serialization.add_safe_globals(sbx_types)  # type: ignore[arg-type]

    # Add string entries for JAX module attributes that can't be added as types
    from stable_baselines3.common.safe_globals import add_safe_globals_str

    # JAX module attributes (needed for JAX array/structure serialization)
    add_safe_globals_str(
        [
            "jax._src.tree_util.none_leaf_registry",
            "jax._src.named_sharding.UnspecifiedValue",
        ]
    )

    # Add all JAX traceback_util functions to the allowlist
    for _name in (
        "_add_call_stack_frames",
        "_add_tracebackhide_to_hidden_frames",
        "_filtering_mode",
        "_ignore_known_hidden_frame",
        "_is_reraiser_frame",
        "_is_under_reraiser",
        "_path_starts_with",
        "_running_under_ipython",
        "api_boundary",
        "cast",
        "filter_traceback",
        "format_exception_only",
        "include_filename",
        "include_frame",
        "register_exclusion",
        "UnfilteredStackTrace",
    ):
        add_safe_globals_str(f"jax._src.traceback_util.{_name}")

    # Add all JAX errors types to the allowlist
    for _name in (
        "ConcretizationTypeError",
        "JaxRuntimeError",
        "JAXIndexError",
        "JAXTypeError",
        "KeyReuseError",
        "NonConcreteBooleanIndexError",
        "SimplifiedTraceback",
        "TracerArrayConversionError",
        "TracerBoolConversionError",
        "TracerIntegerConversionError",
        "UnexpectedTracerError",
    ):
        add_safe_globals_str(f"jax.errors.{_name}")

    # Verify at least one type was registered (sanity check)
    _allowlist = get_safe_globals()
    assert any("sbx" in entry for entry in _allowlist), "sbx types were not registered with the safe allowlist"


# Auto-register on import
_register()
