import torch

from ..models import GFN


def pis(
    gfn: GFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    assert gfn.backprop_through_state is True

    init_state = gfn.generate_initial_state(batch_size)
    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    with torch.enable_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1])

    loss = log_pfs.sum(-1) - log_pbs.sum(-1) - log_reward

    if return_experience:
        return loss.mean(), traj, log_pfs, log_pbs, log_reward
    else:
        return loss.mean()
