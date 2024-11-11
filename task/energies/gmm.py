import torch
import torch.distributions as D
from torch.distributions.mixture_same_family import MixtureSameFamily

from .base_energy import BaseEnergy


class GaussianMixture(BaseEnergy):
    """
    Two dimensional Gaussian mixture distribution with same std deviation.
    """

    logZ_is_available = True
    _ground_truth_logZ = 0.0

    can_sample = True

    def __init__(
        self,
        device: str,
        dim: int,
        mode_list: torch.Tensor,
        scale: float = 1.0,
    ):
        assert dim == 2
        super().__init__(device=device, dim=dim)

        self._make_gmm_distribution(mode_list, scale)

    def _make_gmm_distribution(self, modes: torch.Tensor, scale: float):
        assert modes.ndim == 2 and modes.shape[1] == 2

        num_modes = len(modes)

        comp = D.Independent(
            D.Normal(modes, torch.ones_like(modes) * scale),
            1,
        )
        mix = D.Categorical(torch.ones(num_modes, device=self.device))
        self.gmm = MixtureSameFamily(mix, comp)

    @BaseEnergy._match_device
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.gmm.log_prob(x)

    @BaseEnergy._match_device
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.gmm.log_prob(x)

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return self.gmm.sample((batch_size,))


class HighDimensionalGMM(BaseEnergy):
    """
    High dimensional Gaussian mixture distribution.
    """

    logZ_is_available = True
    _ground_truth_logZ = 0.0

    can_sample = True

    def __init__(
        self,
        device: str,
        dim: int,
        base_gmm,
    ):
        super().__init__(device=device, dim=dim)

        assert dim % 2 == 0

        self.n_gmm = dim // 2
        self.base_gmm = base_gmm(device=device, dim=2)

    @BaseEnergy._match_device
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x)

    @BaseEnergy._match_device
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.ndim

        batch_size = x.shape[:-1]

        x = x.reshape(-1, 2)
        logit = self.base_gmm.log_prob(x).view(*batch_size, self.n_gmm).sum(dim=-1)

        return logit

    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        return self.base_gmm.sample(batch_size * self.n_gmm).view(batch_size, -1)


class GMM1(GaussianMixture):
    def __init__(self, device: str, dim: int):
        mode_list = torch.tensor([(0.0, 0.0)], device=device)

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=1.0,
        )


class GMM9(GaussianMixture):
    def __init__(self, device: str, dim: int, scale: float = 0.5477222):

        mode_list = torch.tensor(
            [(a, b) for a in [-5.0, 0.0, 5.0] for b in [-5.0, 0.0, 5.0]],
            device=device,
        )

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=scale,
        )


class GMM25(GaussianMixture):
    def __init__(self, device: str, dim: int, scale: float = 0.3):

        mode_list = torch.tensor(
            [
                (a, b)
                for a in [-10.0, -5.0, 0.0, 5.0, 10.0]
                for b in [-10.0, -5.0, 0.0, 5.0, 10.0]
            ],
            device=device,
        )

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=scale,
        )


class GMM40(GaussianMixture):
    def __init__(self, device: str, dim: int = 2):

        mode_list = torch.tensor(
            [
                [-0.2995, 21.4577],
                [-32.9218, -29.4376],
                [-15.4062, 10.7263],
                [-0.7925, 31.7156],
                [-3.5498, 10.5845],
                [-12.0885, -7.8626],
                [-38.2139, -26.4913],
                [-16.4889, 1.4817],
                [15.8134, 24.0009],
                [-27.1176, -17.4185],
                [14.5287, 33.2155],
                [-8.2320, 29.9325],
                [-6.4473, 4.2326],
                [36.2190, -37.1068],
                [-25.1815, -10.1266],
                [-15.5920, 34.5600],
                [-25.9272, -18.4133],
                [-27.9456, -37.4624],
                [-23.3496, 34.3839],
                [17.8487, 19.3869],
                [2.1037, -20.5073],
                [6.7674, -37.3478],
                [-28.9026, -20.6212],
                [25.2375, 23.4529],
                [-17.7398, -1.4433],
                [25.5824, 39.7653],
                [15.8753, 5.4037],
                [26.8195, -23.5521],
                [7.4538, -31.0122],
                [-27.7234, -20.6633],
                [18.0989, 16.0864],
                [-23.6941, 12.0843],
                [21.9589, -5.0487],
                [1.5273, 9.2682],
                [24.8151, 38.4078],
                [-30.8249, -14.6588],
                [15.7204, 33.1420],
                [34.8083, 35.2943],
                [7.9606, -34.7833],
                [3.6797, -25.0242],
            ],
            device=device,
        )

        super().__init__(
            device=device,
            mode_list=mode_list,
            dim=dim,
            scale=1.3133,
        )
