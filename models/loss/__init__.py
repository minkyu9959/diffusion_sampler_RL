from .gfn_loss import (
    trajectory_balance_loss,
    vargrad_loss,
    detailed_balance_loss,
    mle_loss,
    annealed_db,
    GFNForwardLossWrapper,
    GFNBackwardLossWrapper,
)
from .pis import pis


from typing import Callable


def get_gfn_forward_loss(
    loss_type: str,
) -> Callable:
    """
    Get forward loss function based on the loss type.
    Returned loss functions have common interface.
    loss_fn(initial_state, gfn, log_reward_fn, exploration_std=None, return_exp=False)
    """
    if loss_type == "tb":
        return GFNForwardLossWrapper(trajectory_balance_loss)

    elif loss_type == "tb-avg":
        return GFNForwardLossWrapper(vargrad_loss)

    elif loss_type == "db":
        return GFNForwardLossWrapper(detailed_balance_loss)

    elif loss_type == "annealed_db":
        return GFNForwardLossWrapper(annealed_db)

    elif loss_type == "pis":
        return pis

    else:
        return Exception("Invalid forward loss type")


def get_gfn_backward_loss(loss_type: str) -> Callable:
    if loss_type == "tb":
        return GFNBackwardLossWrapper(trajectory_balance_loss)

    elif loss_type == "tb-avg":
        return GFNBackwardLossWrapper(vargrad_loss)

    elif loss_type == "annealed_db":
        return GFNBackwardLossWrapper(annealed_db)

    elif loss_type == "mle":
        return GFNBackwardLossWrapper(mle_loss)

    else:
        raise Exception("Invalid backward loss type")
