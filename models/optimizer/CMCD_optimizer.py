import torch

from omegaconf import DictConfig
from ..CMCD import CMCDSampler


def get_CMCD_optimizer(optimizer_cfg: DictConfig, model: CMCDSampler):

    param_groups = [
        {"params": model.time_encoder.parameters()},
        {"params": model.state_encoder.parameters()},
        {"params": model.control_model.parameters()},
    ]

    param_groups += [{"params": model.logZ_ratio, "lr": optimizer_cfg.lr_flow}]

    if optimizer_cfg.use_weight_decay:
        optimizer = torch.optim.Adam(
            param_groups,
            optimizer_cfg.lr_policy,
            weight_decay=optimizer_cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(param_groups, optimizer_cfg.lr_policy)

    return optimizer
