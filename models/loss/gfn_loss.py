import torch
from torch import Tensor


from models import GFN


def trajectory_balance_loss(
    log_prior: Tensor,
    log_pfs: Tensor,
    log_pbs: Tensor,
    log_Z: Tensor,
    log_reward: Tensor,
):
    loss = 0.5 * (
        (log_prior + log_pfs.sum(-1) + log_Z - log_pbs.sum(-1) - log_reward) ** 2
    )
    return loss.mean()


def annealed_db(
    log_prior: torch.Tensor,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    logZ_ratio: torch.Tensor,
    log_reward_t: torch.Tensor,
):
    loss = 0.5 * (
        (log_pfs + logZ_ratio + log_reward_t[:, :-1] - log_pbs - log_reward_t[:, 1:])
        ** 2
    ).sum(-1)

    return loss.mean()


def annealed_db_on_states(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    logZ_ratio: torch.Tensor,
    cur_state_reward: torch.Tensor,
    next_state_reward: torch.Tensor,
):
    loss = 0.5 * (
        (log_pfs + logZ_ratio + cur_state_reward - log_pbs - next_state_reward) ** 2
    )

    return loss.mean()


def annealed_subtb(
    log_prior: torch.Tensor,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    logZ_ratio: torch.Tensor,
    log_reward_t: torch.Tensor,
    annealing_step: int,
):
    subtrajectory_length = log_pfs.size(1) // annealing_step

    log_pfs = log_pfs.view(-1, annealing_step, subtrajectory_length).sum(-1)
    log_pbs = log_pbs.view(-1, annealing_step, subtrajectory_length).sum(-1)

    loss = 0.5 * (
        (log_pfs + logZ_ratio + log_reward_t[:, :-1] - log_reward_t[:, 1:] - log_pbs)
        ** 2
    ).sum(-1)

    return loss.mean()


def vargrad_loss(
    log_prior: Tensor,
    log_pfs: Tensor,
    log_pbs: Tensor,
    log_reward: Tensor,
):
    # Estimate log Z with batch average.
    log_Z = (log_reward + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    loss = log_Z + (log_prior + log_pfs.sum(-1) - log_reward - log_pbs.sum(-1))
    return 0.5 * (loss**2).mean()


def detailed_balance_loss(
    log_prior: Tensor,
    log_pfs: Tensor,
    log_pbs: Tensor,
    log_flows: Tensor,
    log_Z: Tensor,
    log_reward: Tensor,
):
    log_flows[:, -1] = log_reward

    loss = ((log_pfs + log_flows[:, :-1] - log_pbs - log_flows[:, 1:]) ** 2).sum(-1)
    loss += (log_Z + log_prior - log_flows[:, 0]) ** 2

    return (0.5 * loss).mean()


def mle_loss(log_prior: Tensor, log_pfs: torch.Tensor, log_pbs: torch.Tensor):

    loss = -(log_prior + log_pfs.sum(-1))

    return loss.mean()
