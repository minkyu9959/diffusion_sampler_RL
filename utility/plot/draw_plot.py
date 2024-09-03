"""
Plot drawing utility functions. 
These function takes the axes object and the data to be plotted as input, and return the plot object.
"""

from typing import Callable, Optional

import torch
import numpy as np

from matplotlib.axes import Axes
import seaborn as sns


def get_points_on_2D_grid(
    bounds: tuple, grid_width_n_points: int, device: Optional[str] = "cpu"
):
    """
    For points on the grid
    (bounds[0], bounds[1]) x (bounds[0], bounds[1])
    with n points at each side,
    make the list of points on the grid whose shape are (n**2, 2).
    """

    grid_lower_lim, grid_upper_lim = bounds

    x = torch.linspace(
        grid_lower_lim, grid_upper_lim, grid_width_n_points, device=device
    )
    y = torch.linspace(
        grid_lower_lim, grid_upper_lim, grid_width_n_points, device=device
    )

    points = torch.cartesian_prod(x, y)
    return points


def make_2D_meshgrid(bounds: tuple, grid_width_n_points: int):
    """
    For points on the grid
    (bounds[0], bounds[1]) x (bounds[0], bounds[1])
    with n points at each side,
    make meshgrid tensor X, Y whose shape are (n, n).
    """
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
        bounds=bounds, grid_width_n_points=grid_width_n_points, device=device
    )

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


def draw_2D_kde(sample: torch.Tensor, ax: Axes, bounds: tuple):
    sample = sample.cpu().detach()
    return sns.kdeplot(
        x=sample[:, 0], y=sample[:, 1], cmap="Blues", fill=True, ax=ax, clip=bounds
    )


def draw_time_logZ_plot(
    ax: Axes, logZ_t: torch.Tensor, label: str = "Ground truth logZ_t"
):
    line_object = ax.plot(logZ_t.cpu().numpy(), linewidth=3, label=label)

    ax.set_xlabel("Trajectory length")
    ax.set_ylabel("logZ_t")

    return line_object


def draw_energy_histogram(
    ax: Axes, log_reward: torch.Tensor, name=None, bins=40, range=(90, 160)
):
    hist, bins = np.histogram(
        log_reward.detach().cpu().numpy(), bins=bins, range=range, density=True
    )

    ax.set_xlabel("log reward")
    ax.set_ylabel("count")
    ax.grid(True)

    return ax.plot(bins[1:], hist, label=name, linewidth=2)


def draw_vector_field(
    ax: Axes,
    vecfield: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    plotting_bounds: tuple,
    grid_width_n_points: int,
):
    """
    Draw vector field quiver plot on 2D plot.
    Here, given vf must be batch-support version.
    """

    X, Y = make_2D_meshgrid(plotting_bounds, grid_width_n_points // 20)

    points = get_points_on_2D_grid(
        plotting_bounds,
        grid_width_n_points // 20,
        device=device,
    )

    vectors = vecfield(points).detach()

    return ax.quiver(X, Y, vectors[:, 0], vectors[:, 1])


def draw_sample_trajectory_plot(
    ax: Axes,
    trajectory: torch.Tensor,
):
    trajectory = trajectory.cpu().numpy()

    # extract x, y coordinates.
    traj_xs = trajectory[..., 0]
    traj_ys = trajectory[..., 1]

    # Plot each trajectory (maximum ten trajectory will be plotted).
    for traj_x, traj_y, _ in zip(traj_xs, traj_ys, range(0, 10)):
        ax.plot(traj_x, traj_y, color="black")

        final_x = traj_x[-1]
        dx = (traj_x[-1] - traj_x[-2]) / 10

        final_y = traj_y[-1]
        dy = (traj_y[-1] - traj_y[-2]) / 10

        ax.arrow(
            x=final_x,
            y=final_y,
            dx=dx / 10,
            dy=dy / 10,
            head_width=0.1,
            edgecolor="black",
            facecolor="black",
        )

    return ax
