from .gfn import GFN
from .old_gfn import GFN as OldGFN
from .CMCD import CMCDSampler
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


__all__ = [
    "get_model",
    "SamplerModel",
    "GFN",
    "OldGFN",
    "CMCD",
]
