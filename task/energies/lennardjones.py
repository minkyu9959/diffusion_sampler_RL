import logging

import numpy as np
import torch

from .base_energy import BaseEnergy


def lennard_jones_energy(r, eps=1.0, rm=1.0):
    lj = eps * ((rm / r) ** 12 - 2 * (rm / r) ** 6)
    return lj


def distances_from_vectors(r, eps=1e-6):
    """
    Computes the all-distance matrix from given distance vectors.

    Parameters
    ----------
    r : torch.Tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.

    Returns
    -------
    d : torch.Tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()


def tile(a, dim, n_tile):
    """
    Tiles a pytorch tensor along one an arbitrary dimension.

    Parameters
    ----------
    a : PyTorch tensor
        the tensor which is to be tiled
    dim : Integer
        dimension along the tensor is tiled
    n_tile : Integer
        number of tiles

    Returns
    -------
    b : PyTorch tensor
        the tensor with dimension `dim` tiled `n_tile` times
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    )
    order_index = torch.LongTensor(order_index).to(a).long()
    return torch.index_select(a, dim, order_index)


def distance_vectors(x, remove_diagonal=True):
    r"""
    Computes the matrix :math:`r` of all distance vectors between
    given input points where

    :math:`r_{ij} = x_{i} - y_{j}`

    as used in :footcite:`Khler2020EquivariantFE`

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    r : torch.Tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.

    """
    r = tile(x.unsqueeze(2), 2, x.shape[1])
    r = r - r.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


class LennardJonesEnergy(BaseEnergy):
    logZ_is_available = False
    can_sample = False

    def __init__(
        self,
        spatial_dim: int,
        n_particles: int,
        device: str,
        epsilon: float = 1.0,
        min_radius: float = 1.0,
        oscillator: bool = True,
        oscillator_scale: float = 1.0,
        energy_factor: float = 1.0,
    ):
        super().__init__(device=device, dim=spatial_dim * n_particles)

        self.spatial_dim = spatial_dim
        self.n_particles = n_particles

        self.epsilon = epsilon
        self.min_radius = min_radius
        self.oscillator = oscillator
        self.oscillator_scale = oscillator_scale

        self.energy_factor = energy_factor

    def energy(self, x: torch.Tensor):
        assert x.shape[-1] == self.ndim
        batch_shape = x.shape[:-1]

        x = x.view(*batch_shape, self.n_particles, self.spatial_dim)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self.n_particles, self.spatial_dim))
        )

        lj_energies = lennard_jones_energy(dists, self.epsilon, self.min_radius)
        lj_energies = (
            lj_energies.view(*batch_shape, -1).sum(dim=-1) * self.energy_factor
        )

        if self.oscillator:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(dim=(-2, -1)).view(
                *batch_shape
            )
            lj_energies = lj_energies + osc_energies * self.oscillator_scale

        return lj_energies

    def _generate_sample(self, batch_size: int):
        raise NotImplementedError

    def _remove_mean(self, x):
        x = x.view(-1, self.n_particles, self.spatial_dim)
        return x - torch.mean(x, dim=1, keepdim=True)

    def interatomic_dist(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.n_particles, self.spatial_dim)

        # Compute the pairwise interatomic distances
        # removes duplicates and diagonal
        distances = x[:, None, :, :] - x[:, :, None, :]
        distances = distances[
            :,
            torch.triu(torch.ones((self.n_particles, self.n_particles)), diagonal=1)
            == 1,
        ]
        dist = torch.linalg.norm(distances, dim=-1)
        return dist


class LJ13(LennardJonesEnergy):
    can_sample = True

    def __init__(self, device: str):
        super().__init__(
            spatial_dim=3,
            n_particles=13,
            device=device,
        )
        self.approx_sample = torch.tensor(
            np.load(f"task/energies/data/LJ13.npy"),
            device=device,
        )

    def _generate_sample(self, batch_size: int):
        return self.approx_sample[torch.randperm(batch_size)]


class LJ55(LennardJonesEnergy):
    can_sample = True

    def __init__(self, device: str):
        super().__init__(
            spatial_dim=3,
            n_particles=55,
            device=device,
        )
        self.approx_sample = torch.tensor(
            np.load(f"task/energies/data/LJ55.npy"),
            device=device,
        )

    def _generate_sample(self, batch_size: int):
        return self.approx_sample[torch.randperm(batch_size)]
