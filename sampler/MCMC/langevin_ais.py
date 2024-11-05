import torch
import neptune


from omegaconf import DictConfig

from tqdm import tqdm
from argparse import ArgumentParser

from trainer.utils.langevin import one_step_langevin_dynamic

from task import (
    AnnealedDensities,
    AnnealedEnergy,
    BaseEnergy,
    get_energy_function,
    GaussianEnergy,
    Plotter,
)


CORRECT = True


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
            sample, annealed_energy.log_reward, cfg.ld_step, do_correct=CORRECT
        )

    return sample


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num_sample", type=int, required=True)
    parser.add_argument("-K", "--annealing_step", type=int, required=True)
    parser.add_argument("-T", "--end_time", type=int, required=True)
    args = parser.parse_args()

    cfg = DictConfig(
        {
            "num_samples": args.num_sample,
            "num_time_steps": args.annealing_step,
            "ld_step": args.end_time / args.annealing_step,
            "device": "cuda",
            "energy": {
                "_target_": "energy.gmm.GMM25",
                "dim": 2,
            },
        }
    )

    energy = get_energy_function(cfg.energy, device=cfg.device)
    prior = GaussianEnergy(device="cuda", dim=2, std=1.0)
    plotter = Plotter(energy, **cfg.eval.plot)

    sample = annealed_IS_with_langevin(prior, energy, cfg)

    config_postfix = f"N={args.num_sample}-K={args.annealing_step}-T={args.end_time}"
    dir = "corrected" if CORRECT else "uncorrected"

    fig, ax = plotter.make_sample_plot(sample)
    fig.savefig(
        f"results/figure/AIS-Langevin/{dir}/sample-{config_postfix}.pdf",
        bbox_inches="tight",
    )

    fig, ax = plotter.make_energy_histogram(sample)
    fig.savefig(
        f"results/figure/AIS-Langevin/{dir}/energy-{config_postfix}.pdf",
        bbox_inches="tight",
    )

    torch.save(sample, f"results/figure/AIS-Langevin/{dir}/sample-{config_postfix}.pt")
