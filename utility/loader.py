import torch

from typing import Optional

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from .logger import get_logger
from .seed import set_seed

from sampler.gfn.trainer import BaseTrainer

from task import get_energy_by_name, Plotter


CONFIG_PATH = "/home/guest_dyw/diffusion-sampler/configs"


def load_energy_and_plotter(name: str, device: str = "cpu"):
    energy = get_energy_by_name(name, device=device)
    plotter = Plotter(energy)

    return energy, plotter


def load_energy_model_and_config(
    energy_name: str,
    model_name: str,
    overrides: Optional[list] = None,
):
    if overrides is None:
        overrides = []

    with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
        cfg = compose(
            config_name="main.yaml",
            overrides=[
                "~experiment",
                f"energy={energy_name}",
                f"model={model_name}",
                *overrides,
            ],
        )

    set_seed(cfg.seed)

    energy = get_energy_by_name(energy_name, device=cfg.device)

    model = instantiate(
        cfg.model,
        device=cfg.device,
        energy_function=energy,
    ).to(cfg.device)

    return energy, model, cfg


def load_all_from_experiment_path(experiment_path: str, must_init_logger=False):
    config_path = experiment_path + "/.hydra/config.yaml"

    cfg = OmegaConf.load(config_path)

    set_seed(cfg.seed)

    energy = get_energy_by_name(cfg.energy.name, device=cfg.device)

    model = instantiate(
        cfg.model,
        device=cfg.device,
        energy_function=energy,
    ).to(cfg.device)

    model.load_state_dict(torch.load(experiment_path + "/model.pt"))

    plotter = Plotter(energy)

    if must_init_logger:
        logger = get_logger(cfg, "./eval/temp/result")
        logger.detail_log = True
    else:
        logger = None

    trainer: BaseTrainer = instantiate(
        cfg.train.trainer,
        model=model,
        energy_function=energy,
        train_cfg=cfg.train,
        eval_cfg=cfg.eval,
        logger=logger,
    )

    return energy, model, trainer, plotter
