import os
import random

from typing import Optional

import torch
import numpy as np

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

import neptune

from trainer import (
    BaseTrainer,
    check_config_and_set_read_only,
    make_tag,
    set_experiment_output_dir,
)

from energy import BaseEnergy, get_energy_function
from models import get_model


def train(
    cfg: DictConfig,
    model: torch.nn.Module,
    energy_function: BaseEnergy,
    run: Optional[neptune.Run],
):
    trainer: BaseTrainer = instantiate(
        cfg.train.trainer,
        model=model,
        energy_function=energy_function,
        train_cfg=cfg.train,
        eval_cfg=cfg.eval,
    )

    trainer.run = run

    trainer.train()


@hydra.main(version_base="1.3", config_path="./configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    check_config_and_set_read_only(cfg)

    set_experiment_output_dir()

    set_seed(cfg.seed)

    energy_function: BaseEnergy = get_energy_function(cfg)

    model: torch.nn.Module = get_model(cfg, energy_function).to(cfg.device)

    # logger cannot accept OmegaConf object.
    # Convert it to python dictionary.
    cfg_dict = OmegaConf.to_container(cfg)

    run = neptune.init_run(
        project="dywoo1247/Diffusion-sampler", tags=make_tag(cfg), dependencies="infer"
    )

    run["parameters"] = cfg_dict

    train(cfg, model, energy_function, run)

    run.stop()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


if __name__ == "__main__":
    main()
