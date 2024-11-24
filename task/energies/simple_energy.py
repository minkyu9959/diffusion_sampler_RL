import numpy as np

import torch

from .base_energy import BaseEnergy


class GaussianEnergy(BaseEnergy):
    """Guassian energy function with independent noise to each dimension."""

    logZ_is_available = True
    can_sample: bool = True

    def __init__( 
        self,
        device,
        dim: int,
        std: float = 1.0, 
    ): #Modify : std: float = 1e-3 (like Dirac Delta)
        super().__init__(device, dim)

        self.std = std
        self.logvar = np.log(std**2)

        self.log_two_pi = np.log(2 * np.pi)

        self._ground_truth_logZ = (dim / 2) * (self.log_two_pi + self.logvar)

    @property
    def var(self):
        return self.std**2

    @property
    def sigma(self):
        return self.std

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return torch.randn((batch_size, self.ndim), device=self.device) * self.sigma

    def energy(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        return 0.5 * (x**2).sum(-1) / self.var

    def score(self, x: torch.Tensor):
        """
        For simple Gaussian, score can be computed without autograd.
        """
        assert x.shape[-1] == self.ndim
        return -x / self.var

    def log_prob(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        return -self.energy(x) - self._ground_truth_logZ


class UniformEnergy(BaseEnergy):
    logZ_is_available = False
    can_sample = True

    def __init__(self, device, dim, max_support):
        super().__init__(device, dim)
        self.max_support = max_support

    def energy(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        return torch.zeros(*x.shape[:-1], device=x.device)

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return (
            torch.rand((batch_size, self.ndim), device=self.device)
            * (self.max_support * 2)
            - self.max_support
        )

    def score(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        return torch.zeros_like(x)


class DiracDeltaEnergy(BaseEnergy):
    logZ_is_available = False
    can_sample = True

    def energy(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        return torch.zeros(*x.shape[:-1], device=x.device)

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return torch.zeros((batch_size, self.ndim), device=self.device)

    def log_prob(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        assert (x == torch.zeros(self.ndim, device=self.device)).prod()

        return torch.zeros(*x.shape[:-1], device=x.device)

    def score(self, x: torch.Tensor):
        raise NotImplementedError("DiracDeltaEnergy does not have a score function.")
