import torch

from .gfn_loss import (
    trajectory_balance_loss,
    vargrad_loss,
    detailed_balance_loss,
    mle_loss,
    annealed_db,
    annealed_subtb,
    annealed_vargrad,
)

from ..models import GFN, AnnealedGFN


def forward_tb(
    gfn: GFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    log_prior = gfn.get_logprob_initial_state(init_state)

    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    logZ = gfn.learned_logZ
    with torch.no_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

    loss = trajectory_balance_loss(log_prior, log_pfs, log_pbs, logZ, log_reward)

    if return_experience:
        return loss, traj, log_pfs, log_pbs, log_reward
    else:
        return loss


def backward_tb(gfn: GFN, sample: torch.Tensor):
    traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
    log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

    logZ = gfn.learned_logZ
    with torch.no_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

    loss = trajectory_balance_loss(log_prior, log_pfs, log_pbs, logZ, log_reward)

    return loss


def forward_vargrad(
    gfn: GFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    log_prior = gfn.get_logprob_initial_state(init_state)

    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    with torch.no_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

    loss = vargrad_loss(log_prior, log_pfs, log_pbs, log_reward)

    if return_experience:
        return loss, traj, log_pfs, log_pbs, log_reward
    else:
        return loss


def backward_vargrad(
    gfn: GFN,
    sample: torch.Tensor,
):
    traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
    log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

    with torch.no_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

    loss = vargrad_loss(log_prior, log_pfs, log_pbs, log_reward)

    return loss


def forward_db(
    gfn: GFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    log_prior = gfn.get_logprob_initial_state(init_state)

    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    logZ = gfn.learned_logZ
    with torch.no_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

    log_flows = gfn.get_flow_from_trajectory(traj)

    loss = detailed_balance_loss(
        log_prior, log_pfs, log_pbs, log_flows, logZ, log_reward
    )

    if return_experience:
        return loss, traj, log_pfs, log_pbs, log_reward
    else:
        return loss


def backward_db(
    gfn: GFN,
    sample: torch.Tensor,
):
    traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
    log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

    logZ = gfn.learned_logZ
    with torch.no_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

    log_flows = gfn.get_flow_from_trajectory(traj)

    loss = detailed_balance_loss(
        log_prior, log_pfs, log_pbs, log_flows, logZ, log_reward
    )

    return loss


def forward_mle(
    gfn: GFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    log_prior = gfn.get_logprob_initial_state(init_state)

    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    if return_experience:
        with torch.no_grad():
            log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

    loss = mle_loss(log_prior, log_pfs, log_pbs)

    if return_experience:
        return loss, traj, log_pfs, log_pbs, log_reward
    else:
        return loss


def backward_mle(
    gfn: GFN,
    sample: torch.Tensor,
):
    traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
    log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

    loss = mle_loss(log_prior, log_pfs, log_pbs)

    return loss


def forward_annealed_db(
    gfn: AnnealedGFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    log_prior = gfn.get_logprob_initial_state(init_state)

    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    logZ_ratio = gfn.logZ_ratio

    if gfn.annealing_schedule is None:
        times = torch.linspace(0, 1, traj.size(1), device=traj.device)
    else:
        times = gfn.annealing_schedule()

    log_reward_t = -gfn.annealed_energy.energy(times, traj)

    loss = annealed_db(log_prior, log_pfs, log_pbs, logZ_ratio, log_reward_t)

    if return_experience:
        return loss, traj, log_pfs, log_pbs, log_reward_t[:, -1]
    else:
        return loss


def backward_annealed_db(
    gfn: AnnealedGFN,
    sample: torch.Tensor,
):
    traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
    log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

    logZ_ratio = gfn.logZ_ratio

    if gfn.annealing_schedule is None:
        times = torch.linspace(0, 1, traj.size(1), device=traj.device)
    else:
        times = gfn.annealing_schedule()

    times = torch.linspace(0, 1, traj.size(1), device=traj.device)
    log_reward_t = -gfn.annealed_energy.energy(times, traj)

    loss = annealed_db(log_prior, log_pfs, log_pbs, logZ_ratio, log_reward_t)

    return loss


def forward_annealed_subtb(
    gfn: AnnealedGFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    log_prior = gfn.get_logprob_initial_state(init_state)

    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    logZ_ratio = gfn.logZ_ratio

    times = torch.linspace(0, 1, gfn.annealing_step + 1, device=traj.device)
    traj_len_per_annealing = gfn.trajectory_length // gfn.annealing_step
    log_reward_t = -gfn.annealed_energy.energy(
        times, traj[:, 0::traj_len_per_annealing, :]
    )

    loss = annealed_subtb(
        log_prior, log_pfs, log_pbs, logZ_ratio, log_reward_t, gfn.annealing_step
    )

    if return_experience:
        return loss, traj, log_pfs, log_pbs, log_reward_t[:, -1]
    else:
        return loss


def backward_annealed_subtb(
    gfn: AnnealedGFN,
    sample: torch.Tensor,
):
    traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
    log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

    logZ_ratio = gfn.logZ_ratio

    times = torch.linspace(0, 1, gfn.annealing_step + 1, device=traj.device)
    traj_len_per_annealing = gfn.trajectory_length // gfn.annealing_step

    log_reward_t = -gfn.annealed_energy.energy(
        times, traj[:, 0::traj_len_per_annealing, :]
    )

    loss = annealed_subtb(
        log_prior, log_pfs, log_pbs, logZ_ratio, log_reward_t, gfn.annealing_step
    )

    return loss


def forward_annealed_vargrad(
    gfn: AnnealedGFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    log_prior = gfn.get_logprob_initial_state(init_state)

    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule
    )

    if gfn.annealing_schedule is None:
        times = torch.linspace(0, 1, traj.size(1), device=traj.device)
    else:
        times = gfn.annealing_schedule()

    log_reward_t = -gfn.annealed_energy.energy(times, traj)

    loss = annealed_vargrad(log_prior, log_pfs, log_pbs, log_reward_t)

    if return_experience:
        return loss, traj, log_pfs, log_pbs, log_reward_t[:, -1]
    else:
        return loss


def backward_annealed_vargrad(
    gfn: AnnealedGFN,
    sample: torch.Tensor,
):
    traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)
    log_prior = gfn.get_logprob_initial_state(traj[..., 0, :])

    if gfn.annealing_schedule is None:
        times = torch.linspace(0, 1, traj.size(1), device=traj.device)
    else:
        times = gfn.annealing_schedule()

    times = torch.linspace(0, 1, traj.size(1), device=traj.device)
    log_reward_t = -gfn.annealed_energy.energy(times, traj)

    loss = annealed_vargrad(log_prior, log_pfs, log_pbs, log_reward_t)

    return loss
