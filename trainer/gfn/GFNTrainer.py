"""
Train code for GFN with local search buffer + Langevin parametrization
(Sendera et al., 2024, Improved off-policy training of diffusion samplers)
"""

import torch

from omegaconf import DictConfig

from energy import BaseEnergy
from buffer import *

from trainer import BaseTrainer

from .utils.gfn_utils import (
    calculate_subtb_coeff_matrix,
    get_GFN_optimizer,
    get_buffer,
    get_gfn_forward_loss,
    get_gfn_backward_loss,
    get_exploration_std,
)
from .utils.langevin import langevin_dynamics
from .utils.gfn_losses import *


class GFNTrainer(BaseTrainer):
    def initialize(self):
        train_cfg: DictConfig = self.train_cfg

        self.gfn_optimizer = get_GFN_optimizer(train_cfg.optimizer, self.model)

        self.buffer = get_buffer(train_cfg.buffer, self.energy_function)
        self.local_search_buffer = get_buffer(train_cfg.buffer, self.energy_function)

        self.set_training_mode()

    def set_training_mode(self):
        """
        Depending on the training mode, set the training step function and loss function.
        """
        train_cfg = self.train_cfg

        coeff_matrix = calculate_subtb_coeff_matrix(
            train_cfg.subtb_lambda, self.model.trajectory_length
        ).to(train_cfg.device)

        train_mode = train_cfg.train_mode
        if train_mode == "both_ways":
            self._train_step = self.train_from_both_ways
            self.fwd_loss = get_gfn_forward_loss(train_cfg.fwd_loss, coeff_matrix)
            self.bwd_loss = get_gfn_backward_loss(train_cfg.bwd_loss)

        elif train_mode == "fwd":
            self._train_step = self.train_from_forward_trajectory
            self.fwd_loss = get_gfn_forward_loss(train_cfg.fwd_loss, coeff_matrix)

        elif train_mode == "bwd":
            self._train_step = self.train_from_backward_trajectory
            self.bwd_loss = get_gfn_backward_loss(train_cfg.bwd_loss)

        else:
            raise Exception("Invalid training mode")

    def train_step(self) -> float:
        self.model.zero_grad()

        train_cfg: DictConfig = self.train_cfg

        exploration_std = get_exploration_std(
            epoch=self.current_epoch,
            exploratory=train_cfg.exploratory,
            exploration_factor=train_cfg.exploration_factor,
            exploration_wd=train_cfg.exploration_wd,
        )

        loss = self._train_step(exploration_std)

        loss.backward()
        self.gfn_optimizer.step()
        return loss.item()

    def train_from_both_ways(self, exploration_std):
        # For even epoch, train with forward trajectory
        if self.current_epoch % 2 == 0:
            if self.train_cfg.sampling == "buffer":
                loss, states, _, _, log_r = self.train_from_forward_trajectory(
                    exploration_std,
                    return_exp=True,
                )
                self.buffer.add(states[:, -1], log_r)
            else:
                loss = self.train_from_forward_trajectory(
                    exploration_std,
                )

        # For odd epoch, train with backward trajectory
        else:
            loss = self.train_from_backward_trajectory(exploration_std)

        return loss

    def train_from_forward_trajectory(
        self,
        exploration_std,
        return_exp=False,
    ):
        train_cfg = self.train_cfg
        energy = self.energy_function

        init_state = torch.zeros(
            train_cfg.batch_size, energy.data_ndim, device=train_cfg.device
        )

        return self.fwd_loss(
            initial_state=init_state,
            gfn=self.model,
            log_reward_fn=energy.log_reward,
            exploration_std=exploration_std,
            return_exp=return_exp,
        )

    def train_from_backward_trajectory(
        self,
        exploration_std=None,
    ):

        train_cfg = self.train_cfg
        energy: BaseEnergy = self.energy_function
        gfn_model: GFN = self.model

        buffer = self.buffer
        local_search_buffer = self.local_search_buffer

        epoch = self.current_epoch

        if train_cfg.sampling == "sleep_phase":
            samples = gfn_model.sleep_phase_sample(
                train_cfg.batch_size, exploration_std
            ).to(train_cfg.device)
        elif train_cfg.sampling == "energy":
            samples = energy.sample(train_cfg.batch_size, device=train_cfg.device)
        elif train_cfg.sampling == "buffer":
            if train_cfg.get("local_search"):
                if epoch % train_cfg.local_search.ls_cycle < 2:
                    samples, rewards = buffer.sample()
                    local_search_samples, log_r = langevin_dynamics(
                        samples,
                        energy.log_reward,
                        train_cfg.device,
                        train_cfg.local_search,
                    )
                    local_search_buffer.add(local_search_samples, log_r)

                samples, rewards = local_search_buffer.sample()
            else:
                samples, rewards = buffer.sample()

        loss = self.bwd_loss(
            samples, gfn_model, energy.log_reward, exploration_std=exploration_std
        )

        return loss
