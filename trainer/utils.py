import torch

import PIL

import wandb


from omegaconf import DictConfig, OmegaConf

from energy import BaseEnergy


NAME = None


def check_config_and_set_read_only(cfg: DictConfig):
    if cfg.train.train_mode not in ["fwd", "bwd", "both_ways", "sequential"]:
        raise ValueError("Invalid trajectory source")

    if cfg.train.get("local_search") and cfg.train.train_mode == "fwd":
        raise ValueError("Local search cannot be used with foward trajectory training.")

    # From now, config file cannot be modified.
    OmegaConf.set_readonly(cfg, True)


def set_name_from_config(cfg: DictConfig):
    model_name = ""

    if cfg.train.get("exploratory"):
        if cfg.train.exploration_wd:
            model_name = f"exploration_wd_{cfg.train.exploration_factor}_"
        else:
            model_name = f"exploration_{cfg.train.exploration_factor}_"

    if cfg.model.get("langevin_scaler"):
        model_name += "LP_"
        if cfg.model.langevin_scaler.out_dim == cfg.energy.dim:
            model_name += "scaling_per_dimension_"

    if cfg.model.get("backward_model"):
        model_name += f"learn_pb_scale_range_{cfg.model.pb_scale_range}_"

    if cfg.model.get("clipping"):
        model_name += f"clipping_lgv_{cfg.model.lgv_clip}_gfn_{cfg.model.gfn_clip}_"

    if cfg.train.get("fwd_loss"):
        if cfg.train.fwd_loss == "subtb":
            fwd_loss = f"subtb_subtb_lambda_{cfg.train.subtb_lambda}"
        else:
            fwd_loss = cfg.train.fwd_loss

    train_mode = cfg.train.train_mode
    if train_mode == "both_ways":
        ways = f"fwd_bwd/fwd_{fwd_loss}_bwd_{cfg.train.bwd_loss}"
    elif train_mode == "bwd":
        ways = f"bwd/bwd_{cfg.train.bwd_loss}"
    elif train_mode == "fwd":
        ways = f"fwd/fwd_{fwd_loss}"

    if cfg.train.get("local_search"):
        local_serach_cfg = cfg.train.local_search
        buffer_cfg = cfg.train.buffer

        local_search = f"local_search_iter_{local_serach_cfg.max_iter_ls}_"
        local_search += f"burn_{local_serach_cfg.burn_in}_"
        local_search += f"cycle_{local_serach_cfg.ls_cycle}_"
        local_search += f"step_{local_serach_cfg.ld_step}_"
        local_search += f"beta_{buffer_cfg.beta}_"
        local_search += f"rankw_{buffer_cfg.rank_weight}_"
        local_search += f"prioritized_{buffer_cfg.prioritized}"
        ways = f"{ways}/{local_search}"

    name = f"results/{cfg.energy._target_.split('.')[-1]}/"
    name += f"{model_name}gfn/{ways}/"
    name += f"T_{cfg.model.trajectory_length}/tscale_{cfg.model.t_scale}/"
    name += f"lvr_{cfg.model.log_var_range}/seed_{cfg.seed}/"

    global NAME
    NAME = name

    return name


def get_experiment_name():
    return NAME


def fig_to_image(fig):
    fig.canvas.draw()

    return PIL.Image.frombytes(
        "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
    )


def save_model(model: torch.nn.Module, is_final: bool = False):
    final = "_final" if is_final else ""
    torch.save(model.state_dict(), f"{NAME}model{final}.pt")


def make_interpolated_energy(energy_function: BaseEnergy):
    device = energy_function.device
    log_two_pi = torch.log(torch.tensor(2 * torch.pi, device=device))

    @torch.no_grad
    def interpolated_energy_fn(states: torch.Tensor, times: torch.Tensor):
        # Prior is standard normal gaussian.
        # prior_energy = 0.5 * (log_two_pi - (states**2).sum(-1))
        prior_energy = 0.0
        energy = energy_function.energy(states)

        return (1 - times) * prior_energy + times * energy

    return interpolated_energy_fn
