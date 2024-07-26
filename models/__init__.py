from .base_model import SamplerModel

from .GFN import GFN
from .old_gfn import GFN as OldGFN
from .CMCD import CMCDSampler

from .optimizer import get_CMCD_optimizer, get_GFN_optimizer

import torch

from hydra.utils import instantiate
from omegaconf import DictConfig

from energy import BaseEnergy


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
    "CMCDSampler",
    "get_CMCD_optimizer",
    "get_GFN_optimizer",
]
