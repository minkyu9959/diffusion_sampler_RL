from typing import Callable

import torch
from energy.base_energy import BaseEnergy

from .sample_based_metric import compute_all_distribution_distances
from .density_based_metric import compute_all_density_based_metrics


GROUND_TRUTH_SAMPLE = None


def add_prefix_to_dict_key(prefix: str, dict: dict) -> dict:
    return {f"{prefix}/name": dict[name] for name in dict}


def compute_all_metrics(
    model: torch.nn.Module,
    energy_function: BaseEnergy,
    eval_data_size: int = 2000,
    is_final_evaluation=False,
    do_resample=False,
) -> dict:

    # Generate sample for evaluation once and use it repeatedly.
    resample_is_needed = (
        GROUND_TRUTH_SAMPLE is None or GROUND_TRUTH_SAMPLE.size(0) != eval_data_size
    ) and energy_function.can_sample

    if do_resample or resample_is_needed:
        GROUND_TRUTH_SAMPLE = energy_function.sample(batch_size=eval_data_size)

    # Evaluate model
    model.eval()
    generated_sample = model.sample(batch_size=eval_data_size)
    ground_truth_sample = GROUND_TRUTH_SAMPLE

    metrics = dict()

    # Calculate sample based metric if we can sample from the energy function.
    if energy_function.can_sample:
        sample_based_metrics = compute_all_distribution_distances(
            generated_sample, ground_truth_sample
        )
        metrics.update(sample_based_metrics)

    # Calculate density based metric.
    density_based_metrics = compute_all_density_based_metrics(
        model=model,
        energy_function=energy_function,
        generated_sample=generated_sample,
        eval_data_size=eval_data_size,
    )
    metrics.update(density_based_metrics)

    if is_final_evaluation:
        add_prefix_to_dict_key("final_eval/", metrics)
    else:
        add_prefix_to_dict_key("eval/", metrics)

    return metrics
