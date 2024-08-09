from .base_energy import BaseEnergy, HighDimensionalEnergy
from .funnel import Funnel
from .many_well import ManyWell
from .gmm import GaussianMixture, GMM9, GMM25, GMM40

from .annealed_energy import AnnealedEnergy, AnnealedDensities
from .simple_energy import GaussianEnergy, UniformEnergy, DiracDeltaEnergy

from .plotter import Plotter
from .annealed_logZ import estimate_intermediate_logZ, load_logZ_ratio, logZ_to_ratio

from hydra.utils import instantiate
from omegaconf import DictConfig


def get_energy_function(cfg: DictConfig) -> BaseEnergy:
    energy_function = instantiate(cfg.energy, device=cfg.device)

    return energy_function
