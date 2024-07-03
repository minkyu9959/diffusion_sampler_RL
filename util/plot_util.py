from typing import Callable

import torch

from matplotlib.axes import Axes

from .grid import make_2D_meshgrid, get_points_on_2D_grid


def draw_2D_contour(
    ax: Axes,
    log_prob_func: Callable[[torch.Tensor], torch.Tensor],
    bounds: tuple,
    device: str = "cpu",
    grid_width_n_points: int = 200,
    n_contour_levels: int = 50,
    log_prob_min: float = -1000.0,
):
    """
    Plot contours of a log_prob func that is defined on 2D.
    This function returns contour object.

    :Args:
        device (str): device which log_prob_func resides on.
    """

    points = get_points_on_2D_grid(
        bounds=bounds, grid_width_n_points=grid_width_n_points
    ).to(device)

    assert points.ndim == 2 and points.shape[1] == 2

    log_prob_x = log_prob_func(points).detach().cpu()

    log_prob_x = torch.clamp_min(log_prob_x, log_prob_min)

    log_prob_x = log_prob_x.reshape((grid_width_n_points, grid_width_n_points))

    X, Y = make_2D_meshgrid(bounds=bounds, grid_width_n_points=grid_width_n_points)

    return ax.contour(X, Y, log_prob_x, levels=n_contour_levels)


def draw_2D_sample(sample: torch.Tensor, ax: Axes, bounds: tuple, alpha: float = 0.5):
    """
    Draw 2D sample plot.
    This function returns scatter object.
    """
    plot_lower_lim, plot_upper_lim = bounds

    sample = sample.cpu().detach()
    sample = torch.clamp(sample, plot_lower_lim, plot_upper_lim)
    return ax.scatter(sample[:, 0], sample[:, 1], alpha=alpha, marker="o", s=10)
