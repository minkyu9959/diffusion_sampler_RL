from .base_energy import BaseEnergy
from .funnel import Funnel
from .many_well import ManyWell, SymmetricManyWell
from .gmm import (
    GaussianMixture,
    GMM9,
    GMM25,
    GMM40,
    HighDimensionalGMM,
    GMM25ScaledDown,
)
from .double_well import DoubleWellEnergy, DW4
from .lennardjones import LennardJonesEnergy, LJ13, LJ55
from .lgcp import CoxDist as LGCP
from .proteins import AlanineDipeptide

from .annealed_energy import AnnealedEnergy, AnnealedDensities
from .simple_energy import GaussianEnergy, UniformEnergy, DiracDeltaEnergy


def get_energy_by_name(name: str, device: str) -> BaseEnergy:
    match name:
        case "Funnel":
            energy = Funnel(dim=10, device=device)
        case "ManyWell":
            energy = ManyWell(dim=32, device=device)
        case "SymmetricManyWell":
            energy = SymmetricManyWell(dim=32, device=device)
        case "ManyWell128":
            energy = ManyWell(dim=128, device=device)
        case "ManyWell512":
            energy = ManyWell(dim=512, device=device)
        case "GMM9":
            energy = GMM9(dim=2, device=device)
        case "GMM25":
            energy = GMM25(dim=2, device=device)
        case "GMM40":
            energy = GMM40(dim=2, device=device)
        case "MoG50":
            energy = HighDimensionalGMM(dim=50, device=device, base_gmm=GMM25ScaledDown)
        case "MoG200":
            energy = HighDimensionalGMM(
                dim=200, device=device, base_gmm=GMM25ScaledDown
            )
        case "DW4":
            energy = DW4(device=device)
        case "LJ13":
            energy = LJ13(device=device)
        case "LJ55":
            energy = LJ55(device=device)
        case "LGCP":
            energy = LGCP(device=device)
        case _:
            raise ValueError(f"Unknown energy function: {name}")

    energy.name = name
    return energy


__all__ = [
    "BaseEnergy",
    "Funnel",
    "ManyWell",
    "GaussianMixture",
    "GMM9",
    "GMM25",
    "GMM40",
    "HighDimensionalGMM",
    "DoubleWellEnergy",
    "LennardJonesEnergy",
    "AlanineDipeptide",
    "DW4",
    "LJ13",
    "LJ55",
    "LGCP",
    "AnnealedEnergy",
    "AnnealedDensities",
    "GaussianEnergy",
    "UniformEnergy",
    "DiracDeltaEnergy",
    "get_energy_by_name",
]
