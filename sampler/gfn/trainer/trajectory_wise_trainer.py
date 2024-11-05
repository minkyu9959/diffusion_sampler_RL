"""
Train code for GFN with local search buffer + Langevin parametrization
(Sendera et al., 2024, Improved off-policy training of diffusion samplers)
"""

import torch

from ..buffer import get_buffer
from .base_trainer import BaseTrainer
from ..loss import get_forward_loss, get_backward_loss

from .utils.gfn_utils import (
    get_exploration_std,
)
from .utils.langevin import langevin_dynamics


def get_exploration_schedule(train_cfg, epoch: int):
    return get_exploration_std(
        epoch=epoch,
        exploratory=train_cfg.exploratory,
        exploration_factor=train_cfg.exploration_factor,
        exploration_wd=train_cfg.exploration_wd,
    )


class OnPolicyTrainer(BaseTrainer):
    def initialize(self):
        self.loss_fn = get_forward_loss(self.train_cfg.fwd_loss)

    def train_step(self) -> dict:
        self.model.zero_grad()

        loss = self.loss_fn(
            gfn=self.model,
            batch_size=self.train_cfg.batch_size,
            exploration_schedule=get_exploration_schedule(
                self.train_cfg, self.current_epoch
            ),
        )

        loss.backward()
        if self.model.optimizer_cfg.get("max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.model.optimizer_cfg.max_grad_norm
            )

        self.optimizer.step()

        return {"loss": loss.item()}


class OffPolicyTrainer(BaseTrainer):
    def initialize(self):
        self.set_buffer()

        self.fwd_loss_fn = get_forward_loss(self.train_cfg.fwd_loss)
        self.bwd_loss_fn = get_backward_loss(self.train_cfg.bwd_loss)

    def set_buffer(self):
        train_cfg = self.train_cfg
        self.buffer = get_buffer(train_cfg.buffer, self.energy_function)
        self.local_search_buffer = get_buffer(train_cfg.buffer, self.energy_function)

    def train_step(self) -> float:
        self.model.zero_grad()
        exploration_std = get_exploration_schedule(self.train_cfg, self.current_epoch)

        train_cfg = self.train_cfg

        must_train_with_forward = self.must_train_with_forward(self.current_epoch)

        if must_train_with_forward:
            loss, states, _, _, log_r = self.fwd_loss_fn(
                self.model,
                batch_size=train_cfg.batch_size,
                exploration_schedule=exploration_std,
                return_experience=True,
            )

            self.buffer.add(states[:, -1], log_r)

        # For odd epoch, train with backward trajectory
        else:
            samples = self.sample_from_buffer()
            loss = self.bwd_loss_fn(
                self.model,
                samples,
            )

        loss.backward()
        if self.model.optimizer_cfg.get("max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.model.optimizer_cfg.max_grad_norm
            )

        self.optimizer.step()

        return {
            (
                "forward_loss" if must_train_with_forward else "backward_loss"
            ): loss.item()
        }

    def must_train_with_forward(self, epoch: int) -> bool:
        return epoch % (self.train_cfg.num_backward_train_per_forward_train + 1) == 0

    def sample_from_buffer(self):
        train_cfg = self.train_cfg
        epoch = self.current_epoch
        energy = self.energy_function

        if train_cfg.get("local_search"):
            if epoch % train_cfg.local_search.ls_cycle < 2:
                samples, rewards = self.buffer.sample()
                local_search_samples, log_r = langevin_dynamics(
                    samples,
                    energy.log_reward,
                    train_cfg.device,
                    train_cfg.local_search,
                )
                self.local_search_buffer.add(local_search_samples, log_r)

            samples, rewards = self.local_search_buffer.sample()
        else:
            samples, rewards = self.buffer.sample()

        return samples

    def make_plot(self):
        plot_dict = super(OffPolicyTrainer, self).make_plot()

        if self.current_epoch == 0:
            return plot_dict

        samples = self.sample_from_buffer()
        buffer_sample_fig, _ = self.plotter.make_sample_plot(samples)
        buffer_energy_hist, _ = self.plotter.make_energy_histogram(
            samples, name="buffer-sample"
        )

        plot_dict["buffer-sample"] = buffer_sample_fig
        plot_dict["buffer-energy-hist"] = buffer_energy_hist

        return plot_dict


class SampleBasedTrainer(BaseTrainer):
    def initialize(self):
        self.loss_fn = get_backward_loss(self.train_cfg.bwd_loss)

    def train_step(self) -> float:
        self.model.zero_grad()

        train_cfg = self.train_cfg
        energy = self.energy_function

        samples = energy.sample(train_cfg.batch_size, device=train_cfg.device)

        loss = self.loss_fn(self.model, sample=samples)

        loss.backward()
        if self.model.optimizer_cfg.get("max_grad_norm"):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.model.optimizer_cfg.max_grad_norm
            )

        self.optimizer.step()

        return {"loss": loss.item()}


class TwoWayTrainer(BaseTrainer):
    def initialize(self):
        self.fwd_loss_fn = get_forward_loss(self.train_cfg.fwd_loss)
        self.bwd_loss_fn = get_backward_loss(self.train_cfg.bwd_loss)

    def train_step(self) -> float:
        self.model.zero_grad()
        exploration_std = get_exploration_schedule(self.train_cfg, self.current_epoch)

        train_cfg = self.train_cfg

        # For even epoch, train with forward trajectory
        must_train_with_forward = self.current_epoch % 2 == 0

        if must_train_with_forward:
            loss = self.fwd_loss_fn(
                self.model,
                batch_size=train_cfg.batch_size,
                exploration_schedule=exploration_std,
            )

        # For odd epoch, train with backward trajectory from groun truth sample
        else:
            samples = self.energy_function.sample(
                train_cfg.batch_size, device=train_cfg.device
            )

            loss = self.bwd_loss_fn(
                self.model,
                samples,
            )

        loss.backward()
        self.optimizer.step()

        return {
            (
                "forward_loss" if must_train_with_forward else "backward_loss"
            ): loss.item()
        }
