import os
import random

import torch
import numpy as np

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

import wandb

from trainer import (
    BaseTrainer,
    check_config_and_set_read_only,
    make_wandb_tag,
    set_experiment_output_dir,
)

from energy import BaseEnergy, get_energy_function
from models import get_model


def train(cfg: DictConfig, model: torch.nn.Module, energy_function: BaseEnergy):
    trainer: BaseTrainer = instantiate(
        cfg.train.trainer,
        model=model,
        energy_function=energy_function,
        train_cfg=cfg.train,
        eval_cfg=cfg.eval,
    )

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

    # Wandb logging cannot accept OmegaConf object.
    # Convert it to python dictionary.
    cfg_dict = OmegaConf.to_container(cfg)

    if cfg.get("wandb"):
        wandb.init(
            project=cfg.wandb.project,
            entity="dywoo1247",
            config=cfg_dict,
            tags=make_wandb_tag(cfg),
            group=cfg.wandb.get(
                "group",
                type(energy_function).__name__,
                # If you not specified the group name,
                # it will use the name of energy function by defaults.
            ),
        )

    train(cfg, model, energy_function)

    if cfg.get("wandb"):
        wandb.finish()


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
