import torch

from models import GFN


def trajectory_balance_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_Z: torch.Tensor,
    log_reward: torch.Tensor,
):
    loss = 0.5 * ((log_pfs.sum(-1) + log_Z - log_pbs.sum(-1) - log_reward) ** 2)
    return loss.mean()


def detailed_balance_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    log_reward: torch.Tensor,
):
    log_fs[:, -1] = log_reward

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)

    return loss.mean()


def fwd_tb(
    initial_state, gfn: GFN, log_reward_fn, exploration_std=None, return_exp=False
):
    trajectory, log_pfs, log_pbs = gfn.get_forward_trajectory(
        initial_state,
        exploration_schedule=exploration_std,
    )

    log_Z = gfn.get_learned_logZ(trajectory)

    with torch.no_grad():
        log_r = log_reward_fn(trajectory[:, -1]).detach()

    loss = trajectory_balance_loss(log_pfs, log_pbs, log_Z, log_r)

    if return_exp:
        return loss, trajectory, log_pfs, log_pbs, log_r
    else:
        return loss


def bwd_tb(final_states, gfn: GFN, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs = gfn.get_backward_trajectory(
        final_states,
        exploration_schedule=exploration_std,
    )

    log_Z = gfn.get_learned_logZ(states)

    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    loss = trajectory_balance_loss(log_pfs, log_pbs, log_Z, log_r)
    return loss


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


def bwd_tb_avg(final_states, gfn: GFN, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs = gfn.get_backward_trajectory(
        final_states, exploration_schedule=exploration_std
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
        log_reward = log_reward_fn(states[:, -1]).detach()

    loss = detailed_balance_loss(log_pfs, log_pbs, log_fs, log_reward)

    if return_exp:
        return loss, states, log_pfs, log_pbs, log_fs[:, -1]
    else:
        return loss


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


def bwd_mle(final_states, gfn: GFN, log_reward_fn, exploration_std=None):
    states, log_pfs, log_pbs = gfn.get_backward_trajectory(
        final_states, exploration_schedule=exploration_std
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
