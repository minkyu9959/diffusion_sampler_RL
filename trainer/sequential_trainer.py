"""
Train code for Sequential training of GFN.
"""

import torch

import matplotlib.pyplot as plt

from omegaconf import DictConfig

from hydra.utils import instantiate

from models import DoubleGFN

from .buffer import *

from energy import AnnealedDensities
from trainer import BaseTrainer
from utility import Logger


from .utils.gfn_utils import get_exploration_std


def get_exploration_schedule(train_cfg, epoch: int):
    return get_exploration_std(
        epoch=epoch,
        exploratory=train_cfg.exploratory,
        exploration_factor=train_cfg.exploration_factor,
        exploration_wd=train_cfg.exploration_wd,
    )


class SequentialTrainer(BaseTrainer):
    def __init__(
        self,
        model: DoubleGFN,
        energy_function: BaseEnergy,
        train_cfg: DictConfig,
        eval_cfg: DictConfig,
        logger: Logger,
    ):
        super(SequentialTrainer, self).__init__(
            model=model,
            energy_function=energy_function,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            logger=logger,
        )

        subtrainer_cfg = train_cfg.subtrainer
        stage = train_cfg.stage
        model_path = train_cfg.model_path

        # Instantiate first and second GFN trainer
        self.first_gfn_trainer = instantiate(
            subtrainer_cfg.trainer,
            model=self.model.first_gfn,
            energy_function=self.model.intermediate_energy,
            train_cfg=subtrainer_cfg,
            eval_cfg=eval_cfg,
            logger=logger,
            _recursive_=False,
        )

        self.second_gfn_trainer = instantiate(
            subtrainer_cfg.trainer,
            model=self.model.second_gfn,
            energy_function=energy_function,
            train_cfg=subtrainer_cfg,
            eval_cfg=eval_cfg,
            logger=logger,
            _recursive_=False,
        )

        if stage not in [None, 0, 1]:
            raise ValueError("stage must be 0, 1 or None.")
        self._stage = stage

        # load pre-trained first gfn model.
        if stage == 1:
            if model_path is None:
                raise ValueError("model_path must be provided if stage is 1.")
            self.model.load_state_dict(torch.load(model_path))

    def initialize(self):
        self.annealed_energy: AnnealedDensities = self.model.annealed_energy
        self.first_gfn_trainer.initialize()
        self.second_gfn_trainer.initialize()

    @property
    def stage(self) -> int:
        if self._stage is not None:
            return self._stage
        return 0 if (self.current_epoch <= self.max_epoch // 2) else 1

    def train_step(self) -> dict:
        self.model.zero_grad()

        if self.stage == 0:
            return self.first_gfn_trainer.train_step()
        elif self.stage == 1:
            return self.second_gfn_trainer.train_step()

    def eval_step(self) -> dict:
        """
        Execute evaluation step and return metric dictionary.

        Returns:
            metric: a dictionary containing metric value
        """
        if self.stage == 0:
            return self.first_gfn_trainer.eval_step()
        elif self.stage == 1:
            return self.second_gfn_trainer.eval_step()

    def make_plot(self):
        """
        Generate sample from model and plot it using plotter.
        If you want to add more visualization, you can override this method.

        Returns:
            dict: dictionary that has figure objects as value
        """

        if self.stage == 0:
            model = self.model.first_gfn
        elif self.stage == 1:
            model = self.model.second_gfn

        samples = model.sample(batch_size=self.eval_cfg.plot_sample_size)

        sample_fig, _ = self.plotter.make_sample_plot(samples)
        kde_fig, _ = self.plotter.make_kde_plot(samples)

        return {
            "sample-plot": sample_fig,
            "kde-plot": kde_fig,
        }
