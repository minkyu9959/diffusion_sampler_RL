import torch

from energy import BaseEnergy


class AnnealedEnergy:
    def __init__(
        self, energy_function: BaseEnergy, prior: str = "gaussian", log_var: float = 0.0
    ):
        self.energy_function = energy_function
        self.device = energy_function.device
        self.log_two_pi = torch.log(torch.tensor(2 * torch.pi, device=self.device))

        if prior == "gaussian":
            self.log_var = log_var
            self.get_logZ = self.gaussian_logZ
            self.prior_energy = self.gaussian_energy_function
            self.prior_score = self.gaussian_score
        elif prior == "uniform":
            self.prior_energy = self.uniform_energy_function
            self.prior_score = self.uniform_score

    def energy(self, times: torch.Tensor, states: torch.Tensor):
        # Prior is standard normal gaussian.
        prior_energy = self.prior_energy(states)

        energy = self.energy_function.energy(states)

        return (1 - times) * prior_energy + times * energy

    def score(self, times: torch.Tensor, states: torch.Tensor):
        target_score = self.energy_function.score(states)
        prior_score = self.prior_score(states)

        return (1 - times) * prior_score + times * target_score

    def uniform_energy_function(self, states: torch.Tensor):
        return torch.zeros(*states.shape[:-1])

    def uniform_score(self, states: torch.Tensor):
        return torch.zeros(*states.shape[:-1])

    def gaussian_logZ(self):
        return 0.5 * (self.log_two_pi + self.log_var)

    def gaussian_energy_function(self, states: torch.Tensor):
        return 0.5 * (states**2).sum(-1) * torch.exp(-self.log_var)

    def gaussian_score(self, states: torch.Tensor):
        return states * torch.exp(-self.log_var)
