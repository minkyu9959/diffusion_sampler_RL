from typing import Callable

import torch
from energy.base_energy import BaseEnergy

from .sample_based_metric import compute_all_distribution_distances

from models.base_model import SamplerModel

from .density_based_metric import (
    compute_all_density_based_metrics,
    log_partition_function,
)


# Ground truth sample from energy function.
GROUND_TRUTH_SAMPLE = None


def add_prefix_to_dict_key(prefix: str, dict: dict) -> dict:
    return {f"{prefix}/name": dict[name] for name in dict}


def compute_all_metrics(
    model: SamplerModel,
    eval_data_size: int = 2000,
    do_resample=False,
) -> dict:
    global GROUND_TRUTH_SAMPLE

    energy_function = model.energy_function

    # Generate sample for evaluation once and use it repeatedly.
    resample_is_needed = (
        GROUND_TRUTH_SAMPLE is None or GROUND_TRUTH_SAMPLE.size(0) != eval_data_size
    ) and energy_function.can_sample

    if do_resample or resample_is_needed:
        GROUND_TRUTH_SAMPLE = energy_function.sample(batch_size=eval_data_size)

    # Evaluate model
    metrics = dict()

    model.eval()
    generated_sample = model.sample(batch_size=eval_data_size)

    if energy_function.can_sample:
        # Sample based metric if we can sample from the energy function.
        assert GROUND_TRUTH_SAMPLE is not None

        metrics.update(
            compute_all_distribution_distances(generated_sample, GROUND_TRUTH_SAMPLE)
        )

    # Density based metric.
    metrics.update(
        compute_all_density_based_metrics(
            model,
            generated_sample,
        )
    )

    # Estimate log partition value.
    metrics.update(log_partition_function(model, sample_size=eval_data_size))

    model.train()

    return metrics
