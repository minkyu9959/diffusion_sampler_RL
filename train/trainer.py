import abc

import torch

import matplotlib.pyplot as plt
from tqdm import trange
import wandb

from omegaconf import DictConfig

from energy import BaseEnergy
from buffer import *

from metrics import compute_all_metrics, add_prefix_to_dict_key
from train.utils import save_model, get_experiment_name, draw_sample_plot


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
        model: torch.nn.Module,
        energy_function: BaseEnergy,
        train_cfg: DictConfig,
        eval_cfg: DictConfig,
    ):
        self.model = model
        self.energy_function = energy_function
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

        plot_filename_prefix = get_experiment_name()
        plot_sample_size = self.eval_cfg.plot_sample_size
        eval_data_size = (
            self.eval_cfg.final_eval_data_size
            if self.train_end
            else self.eval_cfg.eval_data_size
        )

        metrics = compute_all_metrics(
            model=self.model,
            energy_function=self.energy_function,
            eval_data_size=eval_data_size,
            do_resample=self.train_end,
            # At the end of training, we resample the evaluation data.
        )

        # TODO: Replace them with filter code in compute_all_metircs
        if "tb-avg" in self.train_cfg.mode_fwd or "tb-avg" in self.train_cfg.mode_bwd:
            del metrics["eval/log_Z_learned"]

        if self.train_end:
            add_prefix_to_dict_key("final_eval/", metrics)
        else:
            add_prefix_to_dict_key("eval/", metrics)

        metrics.update(
            draw_sample_plot(
                energy=self.energy_function,
                model=self.model,
                plot_prefix=plot_filename_prefix,
                plot_sample_size=plot_sample_size,
            )
        )

        # Prevent too much plt objects from lasting
        plt.close("all")

        return metrics

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
                wandb.log(metrics, step=epoch)

            if self.must_save(epoch):
                self.save_model()

        # Final evaluation and save model
        metrics.update(self.eval_step())
        self.save_model()

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
