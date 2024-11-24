from .gfn_loss import (
    trajectory_balance_loss,
    vargrad_loss,
    detailed_balance_loss,
    mle_loss,
    annealed_db,
    annealed_db_on_states,
    annealed_subtb,
)

from .trajectory_loss import (
    forward_tb,
    backward_tb,
    forward_vargrad,
    backward_vargrad,
    forward_db,
    backward_db,
    forward_annealed_db,
    backward_annealed_db,
    backward_mle,
    forward_annealed_subtb,
    backward_annealed_subtb,
    forward_annealed_vargrad,
    backward_annealed_vargrad,
)

from .pis import pis
from .jarzynski import jarzynski_reverse_KL


from ..models import SamplerModel
from torch import Tensor

from typing import Callable


def get_forward_loss(
    loss_type: str,
) -> Callable[[SamplerModel, int, Callable, bool], Tensor]:
    """
    Get forward loss function based on the loss type.
    Returned loss functions have common interface.
    loss_fn(model, batch_size, exploration_schedule=None, return_experience=False)
    """
    if loss_type == "tb":
        return forward_tb

    elif loss_type == "tb-avg":
        return forward_vargrad

    elif loss_type == "db":
        return forward_db

    elif loss_type == "adb":
        return forward_annealed_db

    elif loss_type == "asubtb":
        return forward_annealed_subtb

    elif loss_type == "avargrad":
        return forward_annealed_vargrad

    elif loss_type == "pis":
        return pis
    
    elif loss_type == "jarzynski":
        return jarzynski_reverse_KL

    else:
        raise Exception("Invalid forward loss type")


def get_backward_loss(loss_type: str) -> Callable[[SamplerModel, Tensor], Tensor]:
    """
    Get backward loss function based on the loss type.
    Returned loss functions have common interface.
    loss_fn(model: SamplerModel, sample: torch.Tensor)
    """

    if loss_type == "tb":
        return backward_tb

    elif loss_type == "tb-avg":
        return backward_vargrad

    elif loss_type == "adb":
        return backward_annealed_db

    elif loss_type == "asubtb":
        return backward_annealed_subtb

    elif loss_type == "avargrad":
        return backward_annealed_vargrad

    elif loss_type == "mle":
        return backward_mle

    elif loss_type == "db":
        return backward_db

    else:
        raise Exception("Invalid backward loss type")
