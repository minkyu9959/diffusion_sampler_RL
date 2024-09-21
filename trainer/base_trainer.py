import abc

import torch

from tqdm import trange

from omegaconf import DictConfig

from energy import BaseEnergy
from models import SamplerModel, GFN
from utility import Logger, SamplePlotter

from metrics import compute_all_metrics


class BaseTrainer(abc.ABC):
    """
    Base Trainer class for training models.
    User need to implement the following methods:
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
        logger: Logger,
    ):
        self.model = model
        self.energy_function = energy_function
        self.plotter = SamplePlotter(energy_function, **eval_cfg.get("plot", {}))

        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg

        self.current_epoch = 0
        self.max_epoch = train_cfg.epochs

        self.logger = logger

        self.optimizer = self.model.get_optimizer()

    @abc.abstractmethod
    def initialize(self):
        pass

    def train_step(self) -> dict:
        """
        Execute one training step and return train loss.

        Returns:
            loss: training loss
        """
        raise NotImplementedError("train_step method must be implemented.")

    @torch.no_grad()
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

        return compute_all_metrics(
            model=self.model,
            eval_data_size=eval_data_size,
            do_resample=self.train_end,
            # At the end of training, we resample the evaluation data.
        )

    @torch.no_grad()
    def make_plot(self):
        """
        Generate sample from model and plot it using plotter.
        If you want to add more visualization, you can override this method.

        Returns:
            dict: dictionary that has figure objects as value
        """
        sample = self.model.sample(batch_size=self.eval_cfg.plot_sample_size)

        plot_dict = {}

        if self.plotter.can_draw_sample_plot:
            sample_fig, _ = self.plotter.make_sample_plot(sample)
            plot_dict["sample-plot"] = sample_fig

        if self.plotter.can_draw_kde_plot:
            kde_fig, _ = self.plotter.make_kde_plot(sample)
            plot_dict["kde-plot"] = kde_fig

        if self.plotter.can_draw_energy_hist:
            energy_hist_fig, _ = self.plotter.make_energy_histogram(
                sample, name="Model"
            )
            plot_dict["sample-energy-hist"] = energy_hist_fig

        if type(self.model) is not GFN:
            logZ_fig, _ = self.plotter.make_time_logZ_plot(
                self.model.annealed_energy, self.model.logZ_ratio
            )
            plot_dict["logZ-plot"] = logZ_fig

        return plot_dict

    def train(self):
        # Initialize some variables (e.g., optimizer, buffer, frequently used values)
        self.initialize()

        self.model.train()

        for epoch in trange(self.max_epoch + 1):
            self.current_epoch = epoch

            loss = self.train_step()
            self.logger.log_loss(loss, epoch)

            if self.logger.detail_log:
                self.logger.log_gradient(self.model, epoch)

            if self.must_eval(epoch):
                metrics = self.eval_step()
                self.logger.log_metric(metrics, epoch)

                plot = self.make_plot()
                self.logger.log_visual(plot, epoch)

            if self.must_save(epoch):
                self.save_model()

    @property
    def train_end(self):
        return self.current_epoch == self.max_epoch

    def must_save(self, epoch: int = 0):
        return epoch % self.eval_cfg.save_model_every_n_epoch == 0 or self.train_end

    def must_eval(self, epoch: int = 0):
        return epoch % self.eval_cfg.eval_every_n_epoch == 0 or self.train_end

    def save_model(self):
        self.logger.log_model(self.model, self.current_epoch, is_final=self.train_end)
