import torch

import numpy as np

from models import GFN
from buffer import *
from energy import BaseEnergy

from .gfn_losses import *
from omegaconf import DictConfig


def calculate_subtb_coeff_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


def get_GFN_optimizer(
    optimizer_cfg: DictConfig,
    gfn_model: GFN,
):
    param_groups = [
        {"params": gfn_model.time_encoder.parameters()},
        {"params": gfn_model.state_encoder.parameters()},
        {"params": gfn_model.forward_model.parameters()},
    ]
    if gfn_model.langevin_scaler is not None:
        param_groups += [{"params": gfn_model.langevin_scaler.parameters()}]

    if gfn_model.flow_model is not None:
        param_groups += [
            {"params": gfn_model.flow_model.parameters(), "lr": optimizer_cfg.lr_flow}
        ]
    else:
        param_groups += [{"params": gfn_model.logZ, "lr": optimizer_cfg.lr_flow}]

    param_groups += [{"params": gfn_model.logZ_ratio, "lr": optimizer_cfg.lr_flow}]

    if gfn_model.backward_model is not None:
        param_groups += [
            {
                "params": gfn_model.backward_model.parameters(),
                "lr": optimizer_cfg.lr_back,
            }
        ]

    if optimizer_cfg.use_weight_decay:
        gfn_optimizer = torch.optim.Adam(
            param_groups,
            optimizer_cfg.lr_policy,
            weight_decay=optimizer_cfg.weight_decay,
        )
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, optimizer_cfg.lr_policy)

    return gfn_optimizer


def get_buffer(buffer_cfg: DictConfig, energy_function: BaseEnergy) -> BaseBuffer:
    if buffer_cfg.prioritized:
        buffer_class = PrioritizedReplayBuffer
    else:
        buffer_class = SimpleReplayBuffer

    return buffer_class(
        buffer_size=buffer_cfg.buffer_size,
        device=buffer_cfg.device,
        log_reward_function=energy_function.log_reward,
        batch_size=buffer_cfg.batch_size,
        data_ndim=energy_function.data_ndim,
        beta=buffer_cfg.beta,
    )


def get_gfn_forward_loss(
    mode,
    init_state: torch.Tensor,
    gfn_model: GFN,
    log_reward,
    coeff_matrix: torch.Tensor,
    exploration_std=None,
    return_exp=False,
):
    if mode == "tb":
        loss = fwd_tb(
            init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp
        )
    elif mode == "tb-avg":
        loss = fwd_tb_avg(
            init_state, gfn_model, log_reward, exploration_std, return_exp=return_exp
        )
    elif mode == "db":
        loss = db(init_state, gfn_model, log_reward, exploration_std)
    elif mode == "subtb":
        loss = subtb(init_state, gfn_model, log_reward, coeff_matrix, exploration_std)
    elif mode == "annealed-db":
        loss = annealed_db(init_state, gfn_model, log_reward, exploration_std)
    return loss


def get_gfn_backward_loss(mode, samples, gfn_model, log_reward, exploration_std=None):
    if mode == "tb":
        loss = bwd_tb(samples, gfn_model, log_reward, exploration_std)
    elif mode == "tb-avg":
        loss = bwd_tb_avg(samples, gfn_model, log_reward, exploration_std)
    elif mode == "mle":
        loss = bwd_mle(samples, gfn_model, log_reward, exploration_std)
    return loss


def get_exploration_std(
    epoch, exploratory, exploration_factor=0.1, exploration_wd=False
):
    if exploratory is False:
        return None
    if exploration_wd:
        exploration_std = exploration_factor * max(0, 1.0 - epoch / 5000.0)
    else:
        exploration_std = exploration_factor
    expl = lambda x: exploration_std
    return expl
