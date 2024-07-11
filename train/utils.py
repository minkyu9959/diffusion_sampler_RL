import random

import torch

import numpy as np
import PIL

import wandb

from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

from energy import BaseEnergy


NAME = None


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_energy_function(cfg: DictConfig) -> BaseEnergy:

    energy_function = instantiate(cfg.energy, device=cfg.device)

    return energy_function


def get_model(cfg: DictConfig, energy_function: BaseEnergy) -> torch.nn.Module:

    model = instantiate(
        cfg.model,
        dim=energy_function.data_ndim,
        device=cfg.device,
        energy_function=energy_function,
    )

    return model


def get_logger(cfg: DictConfig):
    logger = instantiate(cfg.logger)
    return logger


def add_extra_config_and_set_read_only(cfg: DictConfig):
    if cfg.model.pis_architectures:
        cfg.model.zero_init = True

    if cfg.train.both_ways and cfg.train.bwd:
        cfg.train.bwd = False

    if cfg.train.local_search.do_local_search:
        cfg.train.both_ways = True

    # From now, config file cannot be modified.
    OmegaConf.set_readonly(cfg, True)


def set_name_from_config(cfg: DictConfig):
    name = ""
    if cfg.model.langevin:
        name = "langevin_"
        if cfg.model.langevin_scaling_per_dimension:
            name += "scaling_per_dimension_"

    if cfg.train.exploratory and (cfg.train.exploration_factor is not None):
        if cfg.train.exploration_wd:
            name = f"exploration_wd_{cfg.train.exploration_factor}_{name}_"
        else:
            name = f"exploration_{cfg.train.exploration_factor}_{name}_"

    if cfg.model.learn_pb:
        name = f"{name}learn_pb_scale_range_{cfg.model.pb_scale_range}_"

    if cfg.model.clipping:
        name = f"{name}clipping_lgv_{cfg.model.lgv_clip}_gfn_{cfg.model.gfn_clip}_"

    if cfg.train.mode_fwd == "subtb":
        mode_fwd = f"subtb_subtb_lambda_{cfg.train.subtb_lambda}"
        if cfg.model.partial_energy:
            mode_fwd = f"{mode_fwd}_{cfg.model.partial_energy}"
    else:
        mode_fwd = cfg.train.mode_fwd

    if cfg.train.both_ways:
        ways = f"fwd_bwd/fwd_{mode_fwd}_bwd_{cfg.train.mode_bwd}"
    elif cfg.train.bwd:
        ways = f"bwd/bwd_{cfg.train.mode_bwd}"
    else:
        ways = f"fwd/fwd_{mode_fwd}"

    if cfg.train.local_search.do_local_search:
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

    if cfg.model.pis_architectures:
        results = "results_pis_architectures"
    else:
        results = "results"

    name = f"{results}/{name}gfn/{ways}/T_{cfg.model.trajectory_length}/tscale_{cfg.model.t_scale}/lvr_{cfg.model.log_var_range}/"
    name = f"{results}/{cfg.energy._target_}/{name}gfn/{ways}/T_{cfg.model.trajectory_length}/tscale_{cfg.model.t_scale}/lvr_{cfg.model.log_var_range}/"

    name = f"{name}/seed_{cfg.seed}/"

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


def draw_sample_plot(
    energy: BaseEnergy,
    model: torch.nn.Module,
    plot_prefix: str,
    plot_sample_size: int,
) -> dict:
    """
    Generate sample from model and plot it using energy function's make_plot method.
    If energy function does not have make_plot method, return empty dict.

    Args:
        energy (BaseEnergy): energy function which model learn to sample from
        model (torch.nn.Module): learned sampler model
        plot_prefix (str): plot file prefix (directory path and file name prefix)
        plot_sample_size (int): number of sample to plot

    Returns:
        dict: dictionary that has wandb Image objects as value
    """

    if not hasattr(energy, "make_plot"):
        return {}

    samples = model.sample(batch_size=plot_sample_size)

    fig, _ = energy.make_plot(samples)

    fig.savefig(f"{plot_prefix}plot.pdf", bbox_inches="tight")

    return {
        "visualiation/plot": wandb.Image(fig_to_image(fig)),
    }


def save_model(model: torch.nn.Module, is_final: bool = False):
    final = "_final" if is_final else ""
    torch.save(model.state_dict(), f"{NAME}model{final}.pt")
