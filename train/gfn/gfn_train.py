"""
Train code for GFN with local search buffer + Langevin parametrization
(Sendera et al., 2024, Improved off-policy training of diffusion samplers)
"""

import torch

from omegaconf import DictConfig

from energy import BaseEnergy
from buffer import *

from models import GFN
from train.trainer import BaseTrainer

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


class GFNTrainer(BaseTrainer):
    def initialize(self):
        self.coeff_matrix = calculate_subtb_coeff_matrix(
            self.train_cfg.subtb_lambda, self.model.trajectory_length
        ).to(self.train_cfg.device)

        self.gfn_optimizer = get_GFN_optimizer(self.train_cfg.optimizer, self.model)

        self.buffer = get_buffer(self.train_cfg.buffer, self.energy_function)

        self.local_search_buffer = get_buffer(
            self.train_cfg.buffer, self.energy_function
        )

    def train_step(self) -> float:
        self.model.zero_grad()

        train_cfg: DictConfig = self.train_cfg

        exploration_std = get_exploration_std(
            epoch=self.current_epoch,
            exploratory=train_cfg.exploratory,
            exploration_factor=train_cfg.exploration_factor,
            exploration_wd=train_cfg.exploration_wd,
        )

        if train_cfg.both_ways:
            loss = self._train_from_both_forward_backward_trajectory(exploration_std)
        elif train_cfg.bwd:
            loss = self._train_from_only_backward_trajectory(exploration_std)
        else:
            loss = self._train_from_only_forward_trajectory(exploration_std)

        loss.backward()
        self.gfn_optimizer.step()
        return loss.item()

    def _train_from_both_forward_backward_trajectory(self, exploration_std):
        epoch: int = self.current_epoch

        # For even epoch, train with forward trajectory
        if epoch % 2 == 0:
            if self.train_cfg.sampling == "buffer":
                loss, states, _, _, log_r = fwd_train_step(
                    self.train_cfg,
                    self.energy_function,
                    self.model,
                    exploration_std,
                    self.coeff_matrix,
                    return_exp=True,
                )
                self.buffer.add(states[:, -1], log_r)
            else:
                loss = fwd_train_step(
                    self.train_cfg,
                    self.energy_function,
                    self.model,
                    exploration_std,
                    self.coeff_matrix,
                )

        # For odd epoch, train with backward trajectory
        else:
            loss = bwd_train_step(
                self.train_cfg,
                self.energy_function,
                self.model,
                self.buffer,
                self.local_search_buffer,
                exploration_std,
                it=epoch,
            )
        return loss

    def _train_from_only_backward_trajectory(self, exploration_std):
        return bwd_train_step(
            self.train_cfg,
            self.energy_function,
            self.model,
            self.buffer,
            self.local_search_buffer,
            exploration_std,
            it=self.current_epoch,
        )

    def _train_from_only_forward_trajectory(self, exploration_std):
        return fwd_train_step(
            self.train_cfg,
            self.energy_function,
            self.model,
            exploration_std,
            self.coeff_matrix,
        )


def fwd_train_step(
    train_cfg: DictConfig,
    energy,
    gfn_model,
    exploration_std,
    coeff_matrix,
    return_exp=False,
):
    init_state = torch.zeros(
        train_cfg.batch_size, energy.data_ndim, device=train_cfg.device
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
    energy: BaseEnergy,
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
        samples = energy.sample(train_cfg.batch_size, device=train_cfg.device)
    elif train_cfg.sampling == "buffer":
        if "local_search" in train_cfg and train_cfg.local_search is not None:
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
