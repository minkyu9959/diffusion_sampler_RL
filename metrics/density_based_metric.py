import torch
import math

from energy.base_energy import BaseEnergy
from models.base_model import SamplerModel
from models import GFN


def log_mean_exp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


@torch.no_grad()
def log_partition_function(model: SamplerModel, sample_size: int = 1000):
    """
    Estimate log partition function with sample from trained model.

    Args:
        model: SamplerModel
        sample_size: int
            Number of samples to estimate log partition function.
    """

    log_reward_fn = model.energy_function.log_reward
    init_state = model.generate_initial_state(batch_size=sample_size)

    metrics = {}

    trajectory, log_pfs, log_pbs = model.get_forward_trajectory(init_state)

    if type(model) == GFN:
        # If GFN model and flow is learned (i.e., log Z is learned directly),
        log_Z = model.get_learned_logZ(trajectory)
        metrics["log_Z_learned"] = log_Z.mean()

    sample = trajectory[:, -1]
    log_reward = log_reward_fn(sample)
    log_weight = log_reward + log_pbs.sum(-1) - log_pfs.sum(-1)

    metrics["log_Z_reweighted"] = log_mean_exp(log_weight)

    metrics["log_Z_lower_bound"] = log_weight.mean()

    return metrics


@torch.no_grad()
def estimate_mean_log_likelihood(
    ground_truth_sample: torch.Tensor, model: SamplerModel, num_evals=10
):

    batch_size = ground_truth_sample.shape[0]

    ground_truth_sample = (
        ground_truth_sample.unsqueeze(1)
        .repeat(1, num_evals, 1)
        .view(batch_size * num_evals, -1)
    )

    _, log_pfs, log_pbs = model.get_backward_trajectory(ground_truth_sample)
    log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(batch_size, num_evals, -1)

    return log_mean_exp(log_weight, dim=1).mean()


@torch.no_grad()
def evidence_upper_bound(model: SamplerModel, ground_truth_sample: torch.Tensor):
    """
    Estimate evidence upper bound with ground truth sample.

    Args:
        model: SamplerModel
        ground_truth_sample: torch.Tensor
            Ground truth sample from energy function.
    """

    log_reward_fn = model.energy_function.log_reward

    _, log_pfs, log_pbs = model.get_backward_trajectory(ground_truth_sample)

    log_reward = log_reward_fn(ground_truth_sample)

    log_weight = log_reward + log_pbs.sum(-1) - log_pfs.sum(-1)

    return log_weight.mean()
