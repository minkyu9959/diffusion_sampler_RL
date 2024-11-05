import torch

from .sample_based_metric import distribution_distances

from .density_based_metric import (
    mean_log_likelihood,
    ELBO_and_ELBO_RW,
    EUBO,
)


@torch.no_grad()
def compute_all_metrics(
    model,
    eval_data_size: int = 2000,
    do_resample=False,
) -> dict:

    energy = model.energy_function

    # Evaluate model
    metrics = dict()
    model.eval()

    if energy.can_sample:
        # Use cached sample for efficiency. Only re-generate sample when do_resample=True.
        exact_sample = energy.cached_sample(eval_data_size, do_resample)
        model_sample = model.sample(batch_size=eval_data_size)

        metrics.update(distribution_distances(model_sample, exact_sample))

        # metrics["NLL"] = -mean_log_likelihood(exact_sample, model)

        metrics["EUBO"] = EUBO(model, energy, exact_sample)

    metrics.update(ELBO_and_ELBO_RW(model, energy, sample_size=eval_data_size))

    if hasattr(model, "learned_logZ"):
        metrics["learned_logZ"] = model.learned_logZ.mean()

    model.train()

    return metrics
