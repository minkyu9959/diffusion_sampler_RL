import os

import torch

import hydra
from hydra.utils import call

from omegaconf import DictConfig, OmegaConf

import wandb

from train.utils import (
    set_seed,
    add_extra_config_and_set_read_only,
    get_energy_function,
    get_model,
    get_name_from_config,
)

from energy import BaseEnergy


def train(cfg: DictConfig, model: torch.nn.Module, energy_function: BaseEnergy):
    # model logging

    # energy_funciton logging

    call(
        cfg.train.train_function,
        cfg=cfg,
        model=model,
        energy_function=energy_function,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    add_extra_config_and_set_read_only(cfg)

    set_seed(cfg.seed)

    energy_function: BaseEnergy = get_energy_function(cfg)

    model: torch.nn.Module = get_model(cfg, energy_function).to(cfg.device)

    name = get_name_from_config(cfg)
    if not os.path.exists(name):
        os.makedirs(name)

    cfg_dict = OmegaConf.to_container(cfg)
    wandb.init(project="GFN Energy", entity="dywoo1247", config=cfg_dict, name=name)

    train(cfg, model, energy_function)

    wandb.finish()


if __name__ == "__main__":
    main()
