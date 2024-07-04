import torch
from .util import log_mean_exp

from energy.base_energy import BaseEnergy


@torch.no_grad()
def log_partition_function(model, log_reward_fn):
    initial_state = model.generate_init_state()

    states, log_pfs, log_pbs, log_fs = model.get_trajectory_fwd(
        initial_state, None, log_reward_fn
    )
    log_r = log_reward_fn(states[:, -1])
    log_weight = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)

    log_Z = log_mean_exp(log_weight)
    log_Z_lb = log_weight.mean()
    log_Z_learned = log_fs[:, 0].mean()

    return states[:, -1], log_Z, log_Z_lb, log_Z_learned


@torch.no_grad()
def mean_log_likelihood(data, gfn, log_reward_fn, num_evals=10):
    bsz = data.shape[0]
    data = data.unsqueeze(1).repeat(1, num_evals, 1).view(bsz * num_evals, -1)
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(data, None, log_reward_fn)
    log_weight = (log_pfs.sum(-1) - log_pbs.sum(-1)).view(bsz, num_evals, -1)
    return log_mean_exp(log_weight, dim=1).mean()


def compute_all_density_based_metrics(
    model: torch.nn.Module,
    energy_function: BaseEnergy,
    generated_sample: torch.Tensor,
    eval_data_size: int,
) -> dict:
    metrics = {}

    init_state = model.generate_init_states(eval_data_size)
    (
        _,
        metrics["log_Z"],
        metrics["log_Z_lb"],
        metrics["log_Z_learned"],
    ) = log_partition_function(init_state, model, energy_function.log_reward)

    metrics["mean_log_likelihood"] = mean_log_likelihood(
        generated_sample, model, energy_function.log_reward
    )

    return metrics
