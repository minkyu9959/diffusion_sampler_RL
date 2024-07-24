import abc
import torch

from typing import Optional, Callable


"""
------- Guide for new energy function task implementation -------

Make new energy function as subclass as BaseEnergy.

Then implement the followings:
    1. Set the value of class variable logZ_is_available, can_sample.

    2. _ground_truth_logZ must be set if logZ_is_available.

    3. Implement energy method.
    
    4. Implement _generate_sample method if can_sample.

    5. device and dimension must be set (by BaseEnergy constructor).
"""


class BaseEnergy(abc.ABC):

    logZ_is_available: bool = False
    can_sample: bool = False

    def __init__(self, device, dim):
        self.device = device
        self.data_ndim = dim

    @property
    def ground_truth_logZ(self):
        if not self.logZ_is_available:
            raise Exception("log Z is not available for this energy function")
        return self._ground_truth_logZ

    @staticmethod
    def _match_device(func: Callable[[torch.Tensor], torch.Tensor]):
        def wrapper(self, x: torch.Tensor):
            if x.device != self.device:
                device = x.device
                x = x.to(device=self.device)
            return func(self, x).to(device=device)

        return wrapper

    @abc.abstractmethod
    def energy(self, x: torch.Tensor):
        return

    def unnormalized_density(self, x: torch.Tensor):
        return torch.exp(-self.energy(x))

    @property
    def ndim(self):
        return self.data_ndim

    @abc.abstractmethod
    def _generate_sample(self, batch_size: int) -> torch.Tensor:
        pass

    def sample(self, batch_size: int, device: Optional[str] = None) -> torch.Tensor:
        """
        Generate ground truth sample from energy function.

        Args:
            batch_size (int): Number of sample to generate.

        Returns:
            torch.Tensor: generated sample.
        """

        if not self.can_sample:
            raise Exception(
                "Ground truth sample is not available for this energy function"
            )

        if device is None:
            device = self.device

        return self._generate_sample(batch_size).to(device=device)

    @_match_device
    def score(self, x: torch.Tensor):
        with torch.no_grad():
            copy_x = x.detach().clone()
            copy_x.requires_grad = True
            with torch.enable_grad():
                self.energy(copy_x).sum().backward()
                grad_energy = copy_x.grad.data
            return grad_energy

    def log_reward(self, x: torch.Tensor):
        return -self.energy(x)


class HighDimensionalEnergy(BaseEnergy):
    """
    For high dimensional (d > 2) energy,
    we provide projection and lift to 2D and 2D-projected energy function.
    """

    def __init__(self, device, dim):
        super().__init__(device=device, dim=dim)

    def projection_on_2d(
        self, x: torch.Tensor, first_dim: int, second_dim: int
    ) -> torch.Tensor:

        # Input x must be (..., D) shape tensor
        if x.shape[-1] != self.data_ndim:
            raise Exception("Input tensor has invalid shape")

        is_first_dim_valid = 0 <= first_dim < self.data_ndim
        is_second_dim_valid = 0 <= second_dim < self.data_ndim

        if not is_first_dim_valid or not is_second_dim_valid:
            raise Exception("Invalid projection dimension")

        return torch.stack(
            (x[..., first_dim], x[..., second_dim]), dim=-1, device=x.device
        )

    def lift_from_2d(
        self, projected_x_2d: torch.Tensor, first_dim: int, second_dim: int
    ) -> torch.Tensor:

        # make zero tensor
        x = torch.zeros(
            *projected_x_2d.shape[:-1], self.data_ndim, device=projected_x_2d.device
        )

        x[:, first_dim] = projected_x_2d[..., 0]
        x[:, second_dim] = projected_x_2d[..., 1]

        return x

    def energy_on_2d(
        self, projected_x: torch.Tensor, first_dim: int, second_dim: int
    ) -> torch.Tensor:
        """
        Energy function projected on 2d.
        (
            i.e., energy_on_2d = E compose lift: R^2 -> R^d -> R
            for energy E: R^d -> R
        )

        Args:
            projected_x (torch.Tensor): Batch of 2d points

            first_dim (int):
            The projected point's first dimension corresponds to n-th dimension on whole space.

            second_dim (int):
            The projected point's second dimension corresponds to n-th dimension on whole space.

        Returns:
            torch.Tensor: Batch of energy value
        """

        if projected_x.shape[-1] != 2:
            raise Exception("Input tensor has invalid shape.")

        x = self.lift_from_2d(projected_x, first_dim, second_dim)
        return self.energy(x)
