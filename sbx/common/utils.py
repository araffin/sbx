import optax


def update_learning_rate(opt_state: optax.OptState, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: Optax optimizer state
    :param learning_rate: New learning rate value
    """
    # Note: the optimizer must have been defined with inject_hyperparams
    opt_state.hyperparams["learning_rate"] = learning_rate
