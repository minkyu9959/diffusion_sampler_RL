##############################################
# LogZ_t estimation with importance sampling #
##############################################


import torch

import numpy as np

from energy import BaseEnergy, AnnealedDensities
from energy import GaussianEnergy, GMM9, GMM25, ManyWell, Funnel


def estimate_intermediate_logZ(
    annealed_densities: AnnealedDensities,
    num_samples: int,
    trajectory_length: int,
) -> torch.Tensor:
    """
    Estimate intermediate logZ with importance sampling.

    Args:
        intermediate_energy: AnnealedEnergy
        num_samples: int
            Number of samples to use for importance sampling.
        trajectory_length: int
            Number of step to log Z ratio estimation is performed.

    Returns:
        Tensor: Estimated intermediate logZ's.
    """

    prior_energy = annealed_densities.prior_energy
    target_energy = annealed_densities.energy_function
    target_log_Z = target_energy.ground_truth_logZ
    prior_log_Z = prior_energy.ground_truth_logZ

    def sample_from_importance_distribution(num_samples: int) -> torch.Tensor:
        prior_sample = prior_energy.sample(num_samples)
        target_sample = target_energy.sample(num_samples)

        mask = torch.randint(
            0, 2, (num_samples,), device=prior_sample.device
        ).unsqueeze(-1)

        return prior_sample * mask + target_sample * (1 - mask)

    def importance_distribution_energy(sample: torch.Tensor) -> torch.Tensor:
        sample_prior_energy = prior_energy.energy(sample)
        sample_target_energy = target_energy.energy(sample)

        return -torch.logsumexp(
            torch.stack(
                [
                    -sample_prior_energy + target_log_Z,
                    -sample_target_energy + prior_log_Z,
                ],
            ),
            dim=0,
        )

    sample = sample_from_importance_distribution(num_samples)
    sample_energy = importance_distribution_energy(sample)

    sample_dist_log_Z = target_log_Z + prior_log_Z + np.log(2)

    times = torch.linspace(0.0, 1.0, trajectory_length + 1, device=sample.device)[
        ..., None
    ]

    annealed_energy = annealed_densities.energy(times, sample)

    log_partition = (
        torch.logsumexp(-annealed_energy + sample_energy, dim=1)
        - torch.log(torch.tensor(num_samples))
        + sample_dist_log_Z
    )

    return log_partition


def logZ_to_ratio(log_Z_t: torch.Tensor) -> torch.Tensor:
    """
    Convert logZ to logZ ratio.

    Args:
        logZ: Tensor
            logZ values.
    Returns:
        Tensor: logZ ratio values.
    """
    return log_Z_t[1:] - log_Z_t[:-1]


def load_logZ_ratio(annealed_densities: AnnealedDensities):
    """
    Load ground truth log Z ratio.
    (It's calculated by estimate_intermediate_logZ, i.e., importance sampling.
    But with a large number of samples, so it can be considered as ground truth.)

    Args:
        annealed_densities (AnnealedDensities): annealed density to calculate log Z ratio.

    Returns:
        Tensor: log Z ratio values.
    """

    return torch.load(get_logZ_ratio_filename(annealed_densities))


def get_logZ_ratio_filename(annealed_densities: AnnealedDensities):

    prior_energy_type = type(annealed_densities.prior_energy)
    target_energy_type = type(annealed_densities.energy_function)

    if prior_energy_type == GaussianEnergy:
        prior = f"Gaussian-logvar={annealed_densities.prior_energy.logvar}"
    else:
        raise Exception

    if target_energy_type == GMM9:
        target = "GMM9"
    elif target_energy_type == GMM25:
        target = "GMM25"
    elif target_energy_type == ManyWell:
        target = "ManyWell"
    elif target_energy_type == Funnel:
        target = "Funnel"

    return f"./results/logZ_ratio/{prior}-{target}"
