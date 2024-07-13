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

    trajectories, log_pfs, log_pbs = model.get_forward_trajectory(init_state)
    if type(model) == GFN and model.flow_model is not None:
        # TODO: add aditional model-specfic log Z esitmation metric evaluation code here.
        # If GFN model and flow is learned (i.e., log Z is learned directly),
        trajectories, log_pfs, log_pbs, log_fs = model.get_forward_trajectory(
            init_state, return_log_flow=True
        )
        metrics["log_Z_learned"] = log_fs[:, 0].mean()

    sample = trajectories[:, -1]
    log_reward = log_reward_fn(sample)
    log_weight = log_reward + log_pbs.sum(-1) - log_pfs.sum(-1)

    metrics["log_Z_reweighted"] = log_mean_exp(log_weight)

    metrics["log_Z_lower_bound"] = log_weight.mean()

    return metrics


@torch.no_grad()
def estimate_mean_log_likelihood(
    generated_sample: torch.Tensor, model: SamplerModel, num_evals=10
):

    batch_size = generated_sample.shape[0]

    generated_sample = (
        generated_sample.unsqueeze(1)
        .repeat(1, num_evals, 1)
        .view(batch_size * num_evals, -1)
    )

    _, log_pfs, log_pbs = model.get_backward_trajectory(generated_sample)
    log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(batch_size, num_evals, -1)

    return log_mean_exp(log_weight, dim=1).mean()


@torch.no_grad()
def compute_all_density_based_metrics(
    model: SamplerModel,
    generated_sample: torch.Tensor,
) -> dict:
    metrics = {}

    metrics["estimated_mean_log_likelihood"] = estimate_mean_log_likelihood(
        generated_sample, model
    )

    if model.energy_function.logZ_is_available:
        # Ground truth log prob is available.
        metrics["mean_log_likelihood"] = model.energy_function.log_prob(
            generated_sample
        ).mean()

    return metrics
