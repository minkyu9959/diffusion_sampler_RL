"""
Train code for GFN with local search buffer + Langevin parametrization
(Sendera et al., 2024, Improved off-policy training of diffusion samplers)
"""

import torch

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

from omegaconf import DictConfig

from energy import BaseEnergy
from metrics import compute_all_metrics, add_prefix_to_dict_key
from buffer import *

from models import GFN
from train.utils import draw_sample_plot, get_experiment_name

from .gfn_utils import (
    calculate_subtb_coeff_matrix,
    get_GFN_optimizer,
    get_buffer,
    get_gfn_forward_loss,
    get_gfn_backward_loss,
    get_exploration_std,
)
from .langevin import langevin_dynamics
from .gfn_losses import *


def train_GFlowNets(cfg: DictConfig, model: GFN, energy_function: BaseEnergy):
    assert type(model) == GFN

    train_cfg = cfg.train

    coeff_matrix = calculate_subtb_coeff_matrix(
        train_cfg.subtb_lambda, model.trajectory_length
    ).to(cfg.device)

    gfn_optimizer = get_GFN_optimizer(train_cfg.optimizer, model)

    buffer = get_buffer(train_cfg.buffer, energy_function)
    local_serach_buffer = get_buffer(train_cfg.buffer, energy_function)

    metrics = dict()
    model.train()
    for epoch in trange(train_cfg.epochs + 1):
        metrics["train/loss"] = train_step(
            train_cfg,
            energy_function,
            model,
            gfn_optimizer,
            epoch,
            buffer,
            local_serach_buffer,
            coeff_matrix,
        )

        if must_eval(cfg, epoch):
            eval_step(
                cfg=cfg,
                epoch=epoch,
                metrics=metrics,
                model=model,
                energy_function=energy_function,
            )

        if must_save(cfg, epoch):
            save_model(model)

    # Final evaluation and save model
    eval_step(
        cfg=cfg,
        epoch=epoch,
        metrics=metrics,
        model=model,
        energy_function=energy_function,
        is_final=True,
    )

    save_model(model, is_final=True)


def train_step(
    train_cfg: DictConfig,
    energy: BaseEnergy,
    gfn_model: GFN,
    gfn_optimizer: torch.optim.Optimizer,
    epoch: int,
    buffer: BaseBuffer,
    local_search_buffer: BaseBuffer,
    coeff_matrix: torch.Tensor,
):
    gfn_model.zero_grad()

    exploration_std = get_exploration_std(
        epoch=epoch,
        exploratory=train_cfg.exploratory,
        exploration_factor=train_cfg.exploration_factor,
        exploration_wd=train_cfg.exploration_wd,
    )

    if train_cfg.both_ways:
        if epoch % 2 == 0:
            if train_cfg.sampling == "buffer":
                loss, states, _, _, log_r = fwd_train_step(
                    train_cfg,
                    energy,
                    gfn_model,
                    exploration_std,
                    coeff_matrix,
                    return_exp=True,
                )
                buffer.add(states[:, -1], log_r)
            else:
                loss = fwd_train_step(
                    train_cfg, energy, gfn_model, exploration_std, coeff_matrix
                )
        else:
            loss = bwd_train_step(
                train_cfg,
                energy,
                gfn_model,
                buffer,
                local_search_buffer,
                exploration_std,
                it=epoch,
            )

    elif train_cfg.bwd:
        loss = bwd_train_step(
            train_cfg,
            energy,
            gfn_model,
            buffer,
            local_search_buffer,
            exploration_std,
            it=epoch,
        )

    else:
        loss = fwd_train_step(
            train_cfg, energy, gfn_model, exploration_std, coeff_matrix
        )

    loss.backward()
    gfn_optimizer.step()
    return loss.item()


def fwd_train_step(
    train_cfg: DictConfig,
    energy,
    gfn_model,
    exploration_std,
    coeff_matrix,
    return_exp=False,
):
    init_state = torch.zeros(train_cfg.batch_size, energy.data_ndim).to(
        train_cfg.device
    )
    loss = get_gfn_forward_loss(
        train_cfg.mode_fwd,
        init_state,
        gfn_model,
        energy.log_reward,
        coeff_matrix,
        exploration_std=exploration_std,
        return_exp=return_exp,
    )
    return loss


def bwd_train_step(
    train_cfg: DictConfig,
    energy,
    gfn_model,
    buffer,
    local_search_buffer,
    exploration_std=None,
    it=0,
):
    if train_cfg.sampling == "sleep_phase":
        samples = gfn_model.sleep_phase_sample(
            train_cfg.batch_size, exploration_std
        ).to(train_cfg.device)
    elif train_cfg.sampling == "energy":
        samples = energy.sample(train_cfg.batch_size).to(train_cfg.device)
    elif train_cfg.sampling == "buffer":
        if train_cfg.local_search.do_local_search:
            if it % train_cfg.local_search.ls_cycle < 2:
                samples, rewards = buffer.sample()
                local_search_samples, log_r = langevin_dynamics(
                    samples, energy.log_reward, train_cfg.device, train_cfg.local_search
                )
                local_search_buffer.add(local_search_samples, log_r)

            samples, rewards = local_search_buffer.sample()
        else:
            samples, rewards = buffer.sample()

    loss = get_gfn_backward_loss(
        train_cfg.mode_bwd,
        samples,
        gfn_model,
        energy.log_reward,
        exploration_std=exploration_std,
    )
    return loss


def eval_step(
    cfg: DictConfig,
    epoch: int,
    metrics: dict,
    model: GFN,
    energy_function: BaseEnergy,
    is_final: bool = False,
):

    plot_filename_prefix = get_experiment_name()
    plot_sample_size = cfg.eval.plot_sample_size
    eval_data_size = (
        cfg.eval.final_eval_data_size if is_final else cfg.eval.eval_data_size
    )

    current_metrics = compute_all_metrics(
        model=model,
        energy_function=energy_function,
        eval_data_size=eval_data_size,
        do_resample=is_final,
    )

    if is_final:
        add_prefix_to_dict_key("final_eval/", current_metrics)
    else:
        add_prefix_to_dict_key("eval/", current_metrics)

    metrics.update(current_metrics)

    if "tb-avg" in cfg.train.mode_fwd or "tb-avg" in cfg.train.mode_bwd:
        del metrics["eval/log_Z_learned"]

    metrics.update(
        draw_sample_plot(
            energy_function,
            model,
            plot_filename_prefix,
            plot_sample_size,
        )
    )

    plt.close("all")

    if is_final:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=epoch)


def must_save(cfg: DictConfig, epoch: int = 0):
    return epoch % cfg.eval.save_model_every_n_epoch == 0


def must_eval(cfg: DictConfig, epoch: int = 0):
    return epoch % cfg.eval.eval_every_n_epoch == 0


def save_model(model: GFN, is_final: bool = False):
    name = get_experiment_name()
    final = "_final" if is_final else ""
    torch.save(model.state_dict(), f"{name}model{final}.pt")
