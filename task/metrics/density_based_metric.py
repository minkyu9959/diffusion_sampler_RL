import torch
import math


def log_mean_exp(x, dim=0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def log_importance_weight(log_prior, log_pfs, log_pbs, log_reward):
    """
    log_pfs: (B, T) shaped tensor.
    log_pbs: (B, T) shaped tensor.
    log_reward: B shaped tensor.
    log_prior: B shaped tensor.
    """

    log_pf_traj = log_pfs.sum(-1) + log_prior
    log_pb_traj_given_sample = log_pbs.sum(-1)

    return log_reward + log_pb_traj_given_sample - log_pf_traj


@torch.no_grad()
def ELBO_and_ELBO_RW(model, energy, sample_size=1000):
    init_state = model.generate_initial_state(batch_size=sample_size)
    log_prior = model.get_logprob_initial_state(init_state)
    traj, log_pfs, log_pbs = model.get_forward_trajectory(init_state)

    log_reward = energy.log_reward(traj[:, -1])

    # forward log weight
    log_weight = log_importance_weight(log_prior, log_pfs, log_pbs, log_reward)

    metrics = {}

    metrics["logZ_IS"] = log_mean_exp(log_weight)
    metrics["ELBO"] = log_weight.mean()

    return metrics


@torch.no_grad()
def mean_log_likelihood(exact_sample, model, num_evals=10):

    batch_size = exact_sample.shape[0]

    exact_sample = (
        exact_sample.unsqueeze(1)
        .repeat(1, num_evals, 1)
        .view(batch_size * num_evals, -1)
    )

    trajectory, log_pfs, log_pbs = model.get_backward_trajectory(exact_sample)
    log_prior = model.get_logprob_initial_state(trajectory[..., 0, :])

    log_pf_trajectory = log_prior + log_pfs.sum(-1)
    log_pb_trajectory_given_sample = log_pbs.sum(-1)

    log_weight = (log_pf_trajectory - log_pb_trajectory_given_sample).view(
        batch_size, num_evals, -1
    )

    return log_mean_exp(log_weight, dim=1).mean()


@torch.no_grad()
def EUBO(model, energy, exact_sample):
    """
    Estimate evidence upper bound with ground truth sample.
    """

    traj, log_pfs, log_pbs = model.get_backward_trajectory(exact_sample)

    log_prior = model.get_logprob_initial_state(traj[..., 0, :])

    log_reward = energy.log_reward(exact_sample)

    # backward log weight
    log_weight = log_importance_weight(log_prior, log_pfs, log_pbs, log_reward)

    return log_weight.mean()
