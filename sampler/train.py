import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig, OmegaConf

from task import get_energy_by_name
from utility import get_logger, set_seed


def train(cfg: DictConfig, energy, logger):
    model = instantiate(
        cfg.model,
        device=cfg.device,
        energy_function=energy,
    ).to(cfg.device)

    try:
        trainer = instantiate(
            cfg.train.trainer,
            model=model,
            energy_function=energy,
            train_cfg=cfg.train,
            eval_cfg=cfg.eval,
            logger=logger,
        )

        trainer.train()

    except Exception as e:
        logging.exception(e)

    finally:
        logger.finish()


@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    # From now, the configuration is read-only.
    OmegaConf.set_readonly(cfg, True)

    set_seed(cfg.seed)

    energy = get_energy_by_name(cfg.energy.name, device=cfg.device)

    output_dir = HydraConfig.get().runtime.output_dir
    logger = get_logger(cfg, output_dir)

    train(cfg, energy, logger)


if __name__ == "__main__":
    main()
