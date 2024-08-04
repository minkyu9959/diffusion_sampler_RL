"""
Train code for Sequential training of GFN.
"""

import torch

import matplotlib.pyplot as plt

from omegaconf import DictConfig

from models import DoubleGFN
from models.loss import get_forward_loss, get_backward_loss

from buffer import *

from energy import AnnealedDensities
from trainer import BaseTrainer
from metrics import compute_all_metrics, add_prefix_to_dict_key

from .utils.etc import get_experiment_output_dir


from .utils.gfn_utils import get_exploration_std


def get_exploration_schedule(train_cfg, epoch: int):
    return get_exploration_std(
        epoch=epoch,
        exploratory=train_cfg.exploratory,
        exploration_factor=train_cfg.exploration_factor,
        exploration_wd=train_cfg.exploration_wd,
    )


class SequentialTrainer(BaseTrainer):
    def initialize(self):
        self.annealed_energy: AnnealedDensities = self.model.annealed_energy

        self.first_gfn_optimizer = self.model.first_gfn.get_optimizer()
        self.second_gfn_optimizer = self.model.second_gfn.get_optimizer()

        self.fwd_loss_fn = get_forward_loss(self.train_cfg.fwd_loss)

        # TODO: backward loss function to be implemented.
        # self.bwd_loss_fn = get_backward_loss(self.train_cfg.bwd_loss)

    @property
    def stage(self) -> int:
        return 0 if (self.current_epoch <= self.max_epoch // 2) else 1

    def train_step(self) -> float:
        self.model.zero_grad()

        if self.stage == 0:
            loss = self.fwd_loss_fn(
                self.model.first_gfn,
                batch_size=self.train_cfg.batch_size,
                exploration_schedule=get_exploration_schedule(
                    self.train_cfg, self.current_epoch
                ),
            )

            loss.backward()
            self.first_gfn_optimizer.step()

        elif self.stage == 1:
            stage1_epochs = self.current_epoch - self.max_epoch // 2

            loss = self.fwd_loss_fn(
                self.model.second_gfn,
                batch_size=self.train_cfg.batch_size,
                exploration_schedule=get_exploration_schedule(
                    self.train_cfg, stage1_epochs
                ),
            )

            loss.backward()
            self.second_gfn_optimizer.step()

        return loss.item()

    def eval_step(self) -> dict:
        """
        Execute evaluation step and return metric dictionary.

        Returns:
            metric: a dictionary containing metric value
        """

        eval_data_size = (
            self.eval_cfg.final_eval_data_size
            if self.train_end
            else self.eval_cfg.eval_data_size
        )

        if self.stage == 0:
            model = self.model.first_gfn
        elif self.stage == 1:
            model = self.model.second_gfn

        metrics: dict = compute_all_metrics(
            model=model,
            eval_data_size=eval_data_size,
        )

        if self.train_end:
            metrics = add_prefix_to_dict_key("final_eval/", metrics)
        else:
            metrics = add_prefix_to_dict_key("eval/", metrics)

        return metrics

    def make_plot(self):
        """
        Generate sample from model and plot it using energy function's make_plot method.
        If energy function does not have make_plot method, return empty dict.

        If you want to add more visualization, you can override this method.

        Returns:
            dict: dictionary that has figure objects as value
        """
        output_dir = get_experiment_output_dir()
        plot_sample_size = self.eval_cfg.plot_sample_size

        model = self.model.first_gfn if self.stage == 0 else self.model.second_gfn

        samples = model.sample(batch_size=plot_sample_size)

        fig, _ = self.plotter.make_plot(samples)

        fig.savefig(f"{output_dir}/plot.pdf", bbox_inches="tight")

        return {
            "visuals/sample-plot": fig,
        }
