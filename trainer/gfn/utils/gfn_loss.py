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


def vargrad_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_reward: torch.Tensor,
):
    log_Z = (log_reward + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    loss = log_Z + (log_pfs.sum(-1) - log_reward - log_pbs.sum(-1))
    return 0.5 * (loss**2).mean()


def detailed_balance_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    log_reward: torch.Tensor,
):
    log_fs[:, -1] = log_reward

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)

    return loss.mean()


def mle_loss(log_pfs: torch.Tensor):

    loss = -log_pfs.sum(-1)

    return loss.mean()


def pis(
    gfn: GFN,
    batch_size: int,
    exploration_schedule=None,
    return_experience: bool = False,
):
    init_state = gfn.generate_initial_state(batch_size)
    traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
        init_state, exploration_schedule=exploration_schedule, stochastic_backprop=True
    )

    with torch.enable_grad():
        log_reward = gfn.energy_function.log_reward(traj[:, -1])

    normalization_constant = float(1 / init_state.shape[-1])

    loss = normalization_constant * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_reward)

    if return_experience:
        return loss.mean(), traj, log_pfs, log_pbs, log_reward
    else:
        return loss.mean()


def analyze_loss_fn_argument(loss_fn):
    if loss_fn is trajectory_balance_loss:
        need_log_Z = True
        need_log_reward = True
        need_log_flows = False
    elif loss_fn is vargrad_loss:
        need_log_Z = False
        need_log_reward = True
        need_log_flows = False
    elif loss_fn is detailed_balance_loss:
        need_log_Z = False
        need_log_reward = True
        need_log_flows = True
    elif loss_fn is mle_loss:
        need_log_Z = False
        need_log_reward = False
        need_log_flows = False
    else:
        raise Exception("Not supported loss type.")

    return need_log_Z, need_log_reward, need_log_flows


class GFNForwardLossWrapper:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.need_log_Z, self.need_log_reward, self.need_log_flows = (
            analyze_loss_fn_argument(loss_fn)
        )

    def get_loss(
        self,
        gfn: GFN,
        batch_size: int,
        exploration_schedule=None,
        return_experience: bool = False,
    ):
        init_state = gfn.generate_initial_state(batch_size)
        traj, log_pfs, log_pbs = gfn.get_forward_trajectory(
            init_state, exploration_schedule=exploration_schedule
        )

        kwargs = {
            "log_pfs": log_pfs,
            "log_pbs": log_pbs,
        }

        if self.need_log_Z:
            kwargs["log_Z"] = gfn.get_learned_logZ(traj)

        if self.need_log_reward or return_experience:
            with torch.no_grad():
                log_reward = gfn.energy_function.log_reward(traj[:, -1]).detach()

            if self.need_log_reward:
                kwargs["log_reward"] = log_reward

        if self.need_log_flows:
            kwargs["log_flows"] = gfn.get_flow_from_trajectory(traj)

        loss = self.loss_fn(**kwargs)

        if return_experience:
            return loss, traj, log_pfs, log_pbs, log_reward
        else:
            return loss


class GFNBackwardLossWrapper:
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.need_log_Z, self.need_log_reward, self.need_log_flows = (
            analyze_loss_fn_argument(loss_fn)
        )

    def get_loss(self, gfn: GFN, sample: torch.Tensor):
        traj, log_pfs, log_pbs = gfn.get_backward_trajectory(sample)

        kwargs = {
            "log_pfs": log_pfs,
            "log_pbs": log_pbs,
        }

        if self.need_log_Z:
            kwargs["log_Z"] = gfn.get_learned_logZ(traj)

        if self.need_log_reward:
            kwargs["log_reward"] = gfn.energy_function.log_reward(sample)

        if self.need_log_flows:
            kwargs["log_flows"] = gfn.get_flow_from_trajectory(traj)

        return self.loss_fn(**kwargs)
