import torch

from energy import BaseEnergy


class AnnealedEnergy:
    def __init__(
        self,
        energy_function: BaseEnergy,
        prior_energy: BaseEnergy,
    ):
        self.energy_function = energy_function
        self.device = energy_function.device
        self.prior_energy = prior_energy

    def energy(self, times: torch.Tensor, states: torch.Tensor):
        # Prior is standard normal gaussian.
        prior_energy = self.prior_energy.energy(states)

        energy = self.energy_function.energy(states)

        return (1 - times) * prior_energy + times * energy

    def score(self, times: torch.Tensor, states: torch.Tensor):
        prior_score = self.prior_energy.score(states)

        target_score = self.energy_function.score(states)

        return (1 - times) * prior_score + times * target_score
