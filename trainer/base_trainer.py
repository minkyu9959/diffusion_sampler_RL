import abc

import torch

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

from omegaconf import DictConfig

from energy import BaseEnergy, Plotter
from models import SamplerModel
from buffer import *

from metrics import compute_all_metrics, add_prefix_to_dict_key
from trainer.utils import save_model, get_experiment_name, fig_to_image


class BaseTrainer(abc.ABC):
    """
    Base Trainer class for training models.
    Use need to implement the following methods:
        - initialize
        - train_step

    If you want to use own evaluation method, you can override eval_step method.
    """

    def __init__(
        self,
        model: SamplerModel,
        energy_function: BaseEnergy,
        train_cfg: DictConfig,
        eval_cfg: DictConfig,
    ):
        self.model = model
        self.energy_function = energy_function
        self.plotter = Plotter(energy_function, **eval_cfg.plot)

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        self.current_epoch = 0
        self.max_epoch = train_cfg.epochs

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def train_step(self) -> float:
        """
        Execute one training step and return train loss.

        Returns:
            loss: training loss
        """
        pass

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

        metrics = compute_all_metrics(
            model=self.model,
            eval_data_size=eval_data_size,
            do_resample=self.train_end,
            # At the end of training, we resample the evaluation data.
        )

        if self.train_end:
            metrics = add_prefix_to_dict_key("final_eval/", metrics)
        else:
            metrics = add_prefix_to_dict_key("eval/", metrics)

        metrics.update(self.make_plot())

        # Prevent too much plt objects from lasting
        plt.close("all")

        return metrics

    def make_plot(self):
        """
        Generate sample from model and plot it using energy function's make_plot method.
        If energy function does not have make_plot method, return empty dict.

        If you want to add more visualization, you can override this method.

        Returns:
            dict: dictionary that has wandb Image objects as value
        """
        plot_filename_prefix = get_experiment_name()
        plot_sample_size = self.eval_cfg.plot_sample_size

        model = self.model

        samples = model.sample(batch_size=plot_sample_size)

        fig, _ = self.plotter.make_plot(samples)

        fig.savefig(f"{plot_filename_prefix}plot.pdf", bbox_inches="tight")

        return {
            "visualization/plot": wandb.Image(fig_to_image(fig)),
        }

    def train(self):
        # Initialize some variables (e.g., optimizer, buffer, frequently used values)
        self.initialize()

        # A dictionary to save metric value
        metrics = dict()

        # Traininig
        self.model.train()
        for epoch in trange(self.max_epoch + 1):
            self.current_epoch = epoch

            metrics["train/loss"] = self.train_step()

            if self.must_eval(epoch):
                metrics.update(self.eval_step())

                if wandb.run is not None:
                    wandb.log(metrics, step=epoch)

            if self.must_save(epoch):
                self.save_model()

        # Final evaluation and save model
        metrics.update(self.eval_step())
        self.save_model()

        if wandb.run is not None:
            wandb.log(metrics)

    @property
    def train_end(self):
        return self.current_epoch == self.max_epoch

    def must_save(self, epoch: int = 0):
        return epoch % self.eval_cfg.save_model_every_n_epoch == 0

    def must_eval(self, epoch: int = 0):
        return epoch % self.eval_cfg.eval_every_n_epoch == 0

    def save_model(self):
        save_model(self.model, is_final=self.train_end)
