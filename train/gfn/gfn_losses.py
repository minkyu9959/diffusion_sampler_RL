import torch

from models import GFN


def fwd_tb(
    initial_state, gfn: GFN, log_reward_fn, exploration_std=None, return_exp=False
):
    states, log_pfs, log_pbs = gfn.get_forward_trajectory(
        initial_state,
        exploration_schedule=exploration_std,
    )

    log_Z = gfn.get_learned_logZ(states)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    loss = 0.5 * ((log_pfs.sum(-1) + log_Z - log_pbs.sum(-1) - log_r) ** 2)

    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_r
    else:
        return loss.mean()


def bwd_tb(initial_state, gfn: GFN, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs = gfn.get_backward_trajectory(
        initial_state,
        exploration_schedule=exploration_std,
    )

    log_Z = gfn.get_learned_logZ(states)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    loss = 0.5 * ((log_pfs.sum(-1) + log_Z - log_pbs.sum(-1) - log_r) ** 2)
    return loss.mean()


def fwd_tb_avg(
    initial_state, gfn: GFN, log_reward_fn, exploration_std=None, return_exp=False
):
    states, log_pfs, log_pbs = gfn.get_forward_trajectory(
        initial_state, exploration_schedule=exploration_std
    )
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1))

    if return_exp:
        return 0.5 * (loss**2).mean(), states, log_pfs, log_pbs, log_r
    else:

        return 0.5 * (loss**2).mean()


def bwd_tb_avg(initial_state, gfn: GFN, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs = gfn.get_backward_trajectory(
        initial_state, exploration_schedule=exploration_std
    )
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1))
    return 0.5 * (loss**2).mean()


def db(initial_state, gfn: GFN, log_reward_fn, exploration_std=None, return_exp=False):
    states, log_pfs, log_pbs = gfn.get_forward_trajectory(
        initial_state, exploration_schedule=exploration_std
    )

    log_fs = gfn.get_flow_from_trajectory(states)

    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1]).detach()

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)

    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_fs[:, -1]
    else:
        return loss.mean()


def annealed_db(
    initial_state, gfn: GFN, log_reward_fn, exploration_std=None, return_exp=False
):
    # Generate forward trajectory.
    states, log_pfs, log_pbs = gfn.get_forward_trajectory(
        initial_state, exploration_schedule=exploration_std
    )

    logZ_ratio = gfn.logZ_ratio

    def interpolated_log_reward_fn(states):
        time = torch.linspace(0, 1, gfn.trajectory_length + 1, device=gfn.device)

        # Prior is standard normal gaussian.
        prior_log_reward = -0.5 * (
            torch.log(torch.tensor(2 * torch.pi, device=gfn.device))
            - (states**2).sum(-1)
        )
        log_reward = log_reward_fn(states)

        return (1 - time) * prior_log_reward + time * log_reward

    with torch.no_grad():
        log_r_t = interpolated_log_reward_fn(states)

    loss = 0.5 * (
        (log_pfs + logZ_ratio + log_r_t[:, :-1] - log_pbs - log_r_t[:, 1:]) ** 2
    ).sum(-1)

    if return_exp:
        return loss.mean(), states, log_pfs, log_pbs, log_r_t
    else:
        return loss.mean()


def subtb(
    initial_state,
    gfn: GFN,
    log_reward_fn,
    coef_matrix,
    exploration_std=None,
    return_exp=False,
):
    states, log_pfs, log_pbs = gfn.get_forward_trajectory(
        initial_state, exploration_schedule=exploration_std
    )

    log_fs = gfn.get_flow_from_trajectory(states)

    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1]).detach()

    diff_logp = log_pfs - log_pbs
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp), diff_logp.cumsum(dim=-1)),
        dim=1,
    )
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1
    A2 = A2**2
    if return_exp:
        return (
            torch.stack(
                [
                    torch.triu(A2[i] * coef_matrix, diagonal=1).sum()
                    for i in range(A2.shape[0])
                ]
            ).sum(),
            states,
            log_pfs,
            log_pbs,
            log_fs[:, -1],
        )
    else:

        return torch.stack(
            [
                torch.triu(A2[i] * coef_matrix, diagonal=1).sum()
                for i in range(A2.shape[0])
            ]
        ).sum()


def bwd_mle(samples, gfn: GFN, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs = gfn.get_backward_trajectory(
        samples, exploration_schedule=exploration_std
    )
    loss = -log_pfs.sum(-1)
    return loss.mean()


def pis(initial_state, gfn: GFN, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs = gfn.get_forward_trajectory(
        initial_state, exploration_schedule=exploration_std, stochastic_backprop=True
    )
    with torch.enable_grad():
        log_r = log_reward_fn(states[:, -1])

    normalization_constant = float(1 / initial_state.shape[-1])
    loss = normalization_constant * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_r)
    return loss.mean()
