"""
Train code for Annealed GFN.
"""

import torch
import numpy as np

from models import GFN
from buffer import *

from energy import AnnealedEnergy
from trainer.gfn.GFNTrainer import GFNTrainer

from .utils.langevin import langevin_dynamics


class AnnealedGFNTrainer(GFNTrainer):
    def initialize(self):
        super().initialize()
        self.annealed_energy = AnnealedEnergy(
            self.energy_function, "gaussian", log_var=np.log(9.0)
        )

    def set_training_mode(self):
        """
        Depending on the training mode, set the training step function and loss function.
        """

        train_mode = self.train_cfg.train_mode

        if train_mode == "both_ways":
            self._train_step = self.train_from_both_ways

        elif train_mode == "fwd":
            self._train_step = self.train_from_forward_trajectory

        elif train_mode == "bwd":
            self._train_step = self.train_from_backward_trajectory

        elif train_mode == "sequential":
            self._train_step = self.sequential_training_step

        else:
            raise Exception("Invalid training mode")

    def train_from_forward_trajectory(self, exploration_std, return_exp=False):
        gfn: GFN = self.model

        init_state = gfn.generate_initial_state(self.train_cfg.batch_size)

        trajectory, log_pfs, log_pbs = gfn.get_forward_trajectory(
            init_state, exploration_schedule=exploration_std
        )

        logZ_ratio = gfn.logZ_ratio

        if return_exp:
            loss, log_reward = self.annealed_db(
                trajectory, logZ_ratio, log_pfs, log_pbs, return_reward=True
            )
            return loss, trajectory, log_pfs, log_pbs, log_reward
        else:
            loss = self.annealed_db(
                trajectory, logZ_ratio, log_pfs, log_pbs, return_reward=False
            )
            return loss

    def train_from_backward_trajectory(self, exploration_std):
        train_cfg = self.train_cfg
        energy = self.energy_function
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

        trajectory, log_pfs, log_pbs = gfn_model.get_backward_trajectory(
            samples, exploration_std=exploration_std
        )

        logZ_ratio = gfn_model.logZ_ratio

        loss = self.annealed_db(trajectory, logZ_ratio, log_pfs, log_pbs)

        return loss

    def sequential_training_step(self, exploration_std):
        pass

    def annealed_db(
        self,
        trajectory: torch.Tensor,
        logZ_ratio: torch.Tensor,
        log_pfs: torch.Tensor,
        log_pbs: torch.Tensor,
        return_reward: bool = False,
    ):
        times = torch.linspace(0, 1, trajectory.size(1), device=trajectory.device)

        log_reward_t = -self.annealed_energy.energy(times, trajectory)

        loss = 0.5 * (
            (
                log_pfs
                + logZ_ratio
                + log_reward_t[:, :-1]
                - log_pbs
                - log_reward_t[:, 1:]
            )
            ** 2
        ).sum(-1)

        if return_reward:
            return loss.mean(), log_reward_t[:, -1]
        else:
            return loss.mean()
