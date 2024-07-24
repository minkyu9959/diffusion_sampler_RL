from .gfn import GFN
from .old_gfn import GFN as OldGFN
from .CMCD import CMCD
from .base_model import SamplerModel


from energy import BaseEnergy

from hydra.utils import instantiate
from hydra import compose, initialize

from omegaconf import DictConfig


import torch


def get_model(cfg: DictConfig, energy_function: BaseEnergy) -> SamplerModel:
    model = instantiate(
        cfg.model,
        device=torch.device(cfg.device),
        energy_function=energy_function,
    )

    return model


def get_model_without_config(energy_function: BaseEnergy) -> SamplerModel:
    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(
            config_name="main.yaml",
            overrides=["model=GFN-PIS"],
        )

    return get_model(cfg, energy_function)


__all__ = [
    "get_model",
    "get_model_without_config",
    "SamplerModel",
    "GFN",
    "OldGFN",
    "CMCD",
]
