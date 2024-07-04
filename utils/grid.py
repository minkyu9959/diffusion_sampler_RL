import torch
import itertools


def get_points_on_2D_grid(bounds: tuple, grid_width_n_points: int):
    grid_lower_lim, grid_upper_lim = bounds

    x = torch.linspace(
        grid_lower_lim,
        grid_upper_lim,
        grid_width_n_points,
    )
    y = torch.linspace(
        grid_lower_lim,
        grid_upper_lim,
        grid_width_n_points,
    )

    points = torch.tensor(list(itertools.product(x, y)))

    return points


def make_2D_meshgrid(bounds: tuple, grid_width_n_points: int):
    grid_lower_lim, grid_upper_lim = bounds

    x = torch.linspace(
        grid_lower_lim,
        grid_upper_lim,
        grid_width_n_points,
    )
    y = torch.linspace(
        grid_lower_lim,
        grid_upper_lim,
        grid_width_n_points,
    )

    return torch.meshgrid(x, y, indexing="ij")
