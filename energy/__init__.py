from .base_energy import BaseEnergy, HighDimensionalEnergy
from .funnel import Funnel
from .many_well import ManyWell
from .gmm import GaussianMixture, GMM9, GMM25, GMM40
from .double_well import DoubleWellEnergy
from .lennardjones import LennardJonesEnergy
from .lgcp import CoxDist

from .annealed_energy import AnnealedEnergy, AnnealedDensities
from .simple_energy import GaussianEnergy, UniformEnergy, DiracDeltaEnergy

from hydra.utils import instantiate
from omegaconf import DictConfig


def get_energy_function(cfg: DictConfig, device: str) -> BaseEnergy:
    energy_function = instantiate(cfg, device=device)
    return energy_function


def get_energy_by_name(name: str, device: str) -> BaseEnergy:
    match name:
        case "Funnel":
            energy = Funnel(dim=10, device=device)
        case "ManyWell":
            energy = ManyWell(dim=32, device=device)
        case "ManyWell128":
            energy = ManyWell(dim=128, device=device)
        case "GMM9":
            energy = GMM9(dim=2, device=device)
        case "GMM25":
            energy = GMM25(dim=2, device=device)
        case "GMM40":
            energy = GMM40(dim=2, device=device)
        case "DW4":
            energy = DoubleWellEnergy(spatial_dim=2, n_particles=4, device=device)
        case "LJ13":
            energy = LennardJonesEnergy(spatial_dim=3, n_particles=13, device=device)
        case "LJ55":
            energy = LennardJonesEnergy(spatial_dim=3, n_particles=55, device=device)
        case "LGCP":
            energy = CoxDist(device=device)
        case _:
            raise ValueError(f"Unknown energy function: {name}")

    energy.name = name
    return energy
