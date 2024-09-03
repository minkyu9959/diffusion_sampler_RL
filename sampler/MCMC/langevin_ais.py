import torch
import neptune


from omegaconf import DictConfig

from tqdm import tqdm
from argparse import ArgumentParser

from trainer.utils.langevin import one_step_langevin_dynamic
from energy import (
    AnnealedDensities,
    AnnealedEnergy,
    BaseEnergy,
    get_energy_function,
    GaussianEnergy,
)
from utility import SamplePlotter


@torch.no_grad()
def annealed_IS_with_langevin(prior: BaseEnergy, target: BaseEnergy, cfg: DictConfig):
    annealed_densities = AnnealedDensities(target, prior)

    device = cfg.device
    num_time_steps = cfg.num_time_steps
    num_samples = cfg.num_samples

    sample = prior.sample(num_samples, device)

    for t in tqdm(torch.linspace(0, 1, num_time_steps)[1:]):
        annealed_energy = AnnealedEnergy(annealed_densities, t)
        sample = one_step_langevin_dynamic(
            sample, annealed_energy.log_reward, cfg.ld_step, do_correct=True
        )

    return sample


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num_sample", type=int, required=True)
    parser.add_argument("-A", "--annealing_step", type=int, required=True)
    parser.add_argument("-T", "--MCMC_step", type=int, required=True)
    args = parser.parse_args()

    cfg = DictConfig(
        {
            "num_samples": args.num_sample,
            "num_time_steps": args.annealing_step,
            "max_iter_ls": args.MCMC_step,
            "burn_in": args.MCMC_step - 100,
            "ld_schedule": True,
            "ld_step": 0.01,
            "target_acceptance_rate": 0.574,
            "device": "cuda",
            "energy": {
                "_target_": "energy.many_well.ManyWell",
                "dim": 32,
            },
            "eval": {
                "plot": {
                    "plotting_bounds": [-3.0, 3.0],
                    "projection_dims": [[0, 2], [1, 2], [2, 4], [3, 4], [4, 6], [5, 6]],
                    "fig_size": [12, 20],
                }
            },
        }
    )

    energy = get_energy_function(cfg)
    prior = GaussianEnergy(device="cuda", dim=32, std=1.0)
    plotter = SamplePlotter(energy, **cfg.eval.plot)

    sample = annealed_IS_with_langevin(prior, energy, cfg)

    config_postfix = f"N={args.num_sample}-A={args.annealing_step}-T={args.MCMC_step}"

    fig, ax = plotter.make_sample_plot(sample)
    fig.savefig(
        f"results/figure/AIS-Langevin/sample-{config_postfix}.pdf",
        bbox_inches="tight",
    )

    fig, ax = plotter.make_energy_histogram(sample)
    fig.savefig(
        f"results/figure/AIS-Langevin/energy-{config_postfix}.pdf",
        bbox_inches="tight",
    )
