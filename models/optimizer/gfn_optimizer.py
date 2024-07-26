import torch


from omegaconf import DictConfig
from ..GFN import GFN


def get_GFN_optimizer(
    optimizer_cfg: DictConfig,
    gfn_model: GFN,
):
    param_groups = [
        {"params": gfn_model.time_encoder.parameters()},
        {"params": gfn_model.state_encoder.parameters()},
        {"params": gfn_model.forward_model.parameters()},
    ]

    if gfn_model.backward_model is not None:
        param_groups += [
            {
                "params": gfn_model.backward_model.parameters(),
                "lr": optimizer_cfg.lr_back,
            }
        ]

    if gfn_model.langevin_scaler is not None:
        param_groups += [{"params": gfn_model.langevin_scaler.parameters()}]

    if gfn_model.flow_model is not None:
        param_groups += [
            {"params": gfn_model.flow_model.parameters(), "lr": optimizer_cfg.lr_flow}
        ]
    else:
        # Even though there is no (conditional) flow model, GFN still learn logZ.
        param_groups += [{"params": gfn_model.logZ, "lr": optimizer_cfg.lr_flow}]

    param_groups += [{"params": gfn_model.logZ_ratio, "lr": optimizer_cfg.lr_flow}]

    if optimizer_cfg.use_weight_decay:
        gfn_optimizer = torch.optim.Adam(
            param_groups,
            optimizer_cfg.lr_policy,
            weight_decay=optimizer_cfg.weight_decay,
        )
    else:
        gfn_optimizer = torch.optim.Adam(param_groups, optimizer_cfg.lr_policy)

    return gfn_optimizer
