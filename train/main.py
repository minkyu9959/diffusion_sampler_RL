import os

import torch

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

import wandb

from train.utils import (
    set_seed,
    check_config_and_set_read_only,
    get_energy_function,
    get_model,
    set_name_from_config,
)

from energy import BaseEnergy
from train.trainer import BaseTrainer


def train(cfg: DictConfig, model: torch.nn.Module, energy_function: BaseEnergy):

    # Instantiate the trainer.
    trainer = instantiate(
        cfg.train.trainer,
        model=model,
        energy_function=energy_function,
        train_cfg=cfg.train,
        eval_cfg=cfg.eval,
    )

    trainer.train()


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    check_config_and_set_read_only(cfg)

    set_seed(cfg.seed)

    energy_function: BaseEnergy = get_energy_function(cfg)

    model: torch.nn.Module = get_model(cfg, energy_function).to(cfg.device)

    experiment_name = set_name_from_config(cfg)
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Wandb logging cannot accept OmegaConf object.
    # Convert it to python dictionary.
    cfg_dict = OmegaConf.to_container(cfg)

    wandb.init(
        project=cfg.wandb.project,
        entity="dywoo1247",
        config=cfg_dict,
        name=experiment_name,
    )

    # Train strats here.
    train(cfg, model, energy_function)

    wandb.finish()


if __name__ == "__main__":
    main()
