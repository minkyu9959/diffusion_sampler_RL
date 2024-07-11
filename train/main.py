import os

import torch

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

import wandb

from train.utils import (
    set_seed,
    add_extra_config_and_set_read_only,
    get_energy_function,
    get_model,
    set_name_from_config,
)

from energy import BaseEnergy
from train.trainer import BaseTrainer


def train(cfg: DictConfig, model: torch.nn.Module, energy_function: BaseEnergy):
    # TODO: model logging

    # TODO: energy_funciton logging

    trainer = instantiate(
        cfg.train.trainer,
        model=model,
        energy_function=energy_function,
        train_cfg=cfg.train,
        eval_cfg=cfg.eval,
    )

    trainer.initialize()

    trainer.train()


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

    name = set_name_from_config(cfg)
    if not os.path.exists(name):
        os.makedirs(name)

    # Wandb logging only cannot accept OmegaConf object.
    # Thus, we convert it to python dictionary.
    cfg_dict = OmegaConf.to_container(cfg)
    wandb.init(
        project=cfg.wandb.project, entity="dywoo1247", config=cfg_dict, name=name
    )

    # Train strats here.
    train(cfg, model, energy_function)

    wandb.finish()


if __name__ == "__main__":
    main()
