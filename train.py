import torch

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig

from energy import BaseEnergy, get_energy_by_name
from models import get_model

from trainer import BaseTrainer
from utility import get_logger, Logger, set_seed

from configs.util import *


def train(
    cfg: DictConfig, model: torch.nn.Module, energy_function: BaseEnergy, logger: Logger
):
    trainer: BaseTrainer = instantiate(
        cfg.train.trainer,
        model=model,
        energy_function=energy_function,
        train_cfg=cfg.train,
        eval_cfg=cfg.eval,
        logger=logger,
    )

    trainer.train()


@hydra.main(version_base="1.3", config_path="./configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    check_config_and_set_read_only(cfg)

    set_seed(cfg.seed)

    energy_function: BaseEnergy = get_energy_by_name(cfg.energy.name, device=cfg.device)

    model: torch.nn.Module = get_model(cfg, energy_function).to(cfg.device)

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    logger: Logger = get_logger(cfg, output_dir, debug=cfg.get("detail_log", False))

    train(cfg, model, energy_function, logger)


if __name__ == "__main__":
    main()
