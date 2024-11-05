import torch

from ..energies import BaseEnergy


class Projector:
    def __init__(self, energy: BaseEnergy):
        assert energy.data_ndim > 2, "Energy function must be high-dimensional"

        self.energy = energy
        self.lift_dim = energy.data_ndim

    def projection(
        self, x: torch.Tensor, first_dim: int, second_dim: int
    ) -> torch.Tensor:
        """
        Project batch of high-dimensoinal vector to 2d vector.

        Args:
            x (torch.Tensor): (..., D) shape tensor
            first_dim (int): First projected dimension
            second_dim (int): Second projected dimension

        Returns:
            torch.Tensor: (..., 2) shape tensor
        """
        return x[..., (first_dim, second_dim)]

    def lift(self, x: torch.Tensor, first_dim: int, second_dim: int) -> torch.Tensor:
        """
        Project batch of high-dimensoinal vector to 2d vector.

        Args:
            x (torch.Tensor): (..., 2) shape tensor
            first_dim (int): First projected dimension
            second_dim (int): Second projected dimension

        Returns:
            torch.Tensor: (..., D) shape tensor
        """

        batch_size = x.shape[:-1]
        device = x.device

        x_lifted = torch.zeros(*batch_size, self.lift_dim, device=device)
        x_lifted[..., (first_dim, second_dim)] = x

        return x_lifted

    def energy_on_2d(
        self, x: torch.Tensor, first_dim: int, second_dim: int
    ) -> torch.Tensor:
        """
        Energy function projected on 2d.
        (
            i.e., energy_on_2d = E compose lift: R^2 -> R^d -> R
            for energy E: R^d -> R
        )

        Args:
            x (torch.Tensor): Batch of 2d points

            first_dim (int):
            The projected point's first dimension corresponds to n-th dimension on whole space.

            second_dim (int):
            The projected point's second dimension corresponds to n-th dimension on whole space.

        Returns:
            torch.Tensor: Batch of energy value
        """

        if x.shape[-1] != 2:
            raise Exception("Input tensor has invalid shape.")

        lifted_x = self.lift(x, first_dim, second_dim)
        return self.energy.energy(lifted_x)
