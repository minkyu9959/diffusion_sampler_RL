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
    bounds: tuple, grid_width: int, device: Optional[str] = "cpu"
):
    """
    Generate a list of points on a 2D grid defined by the given bounds.
    The grid will have grid_width points along each axis,
    resulting in a total of (grid_width ** 2) points.
    """

    x_lower_lim, x_upper_lim, y_lower_lim, y_upper_lim = bounds

    x = torch.linspace(x_lower_lim, x_upper_lim, grid_width, device=device)
    y = torch.linspace(y_lower_lim, y_upper_lim, grid_width, device=device)

    points = torch.cartesian_prod(x, y)
    return points


def make_2D_meshgrid(bounds: tuple, grid_width: int):
    """
    Create a meshgrid tensor X, Y with shape (n, n) for points on the grid
    defined by (bounds[0], bounds[1]) x (bounds[2], bounds[3]) with n points on each side.
    """
    x_lower_lim, x_upper_lim, y_lower_lim, y_upper_lim = bounds

    x = torch.linspace(
        x_lower_lim,
        x_upper_lim,
        grid_width,
    )
    y = torch.linspace(
        y_lower_lim,
        y_upper_lim,
        grid_width,
    )

    return torch.meshgrid(x, y, indexing="ij")


def draw_2D_contour(
    ax: Axes,
    func: Callable[[torch.Tensor], torch.Tensor],
    plotting_bounds: tuple,
    grid_width: int = 200,
    n_contour_levels: int = 50,
    func_min_value: float = -1000.0,
    fill_color: bool = False,
    device: str = "cpu",
):
    """
    Plot contours of a func that is defined on 2D.
    This function returns contour object.

    :Args:
        device (str): device which function resides on.
    """

    points = get_points_on_2D_grid(
        bounds=plotting_bounds, grid_width=grid_width, device=device
    )

    assert points.ndim == 2 and points.shape[1] == 2

    z = func(points).detach().cpu()
    z = torch.clamp_min(z, func_min_value)
    z = z.reshape((grid_width, grid_width))

    X, Y = make_2D_meshgrid(bounds=plotting_bounds, grid_width=grid_width)

    if fill_color:
        # Fill the regions outside the contour levels with a dark color
        contour = ax.contourf(
            X,
            Y,
            z,
            levels=n_contour_levels,
        )
    else:
        contour = ax.contour(X, Y, z, levels=n_contour_levels)

    ax.set_xlim(plotting_bounds[0], plotting_bounds[1])
    ax.set_ylim(plotting_bounds[2], plotting_bounds[3])

    return contour


def draw_2D_sample(
    ax: Axes,
    sample: torch.Tensor,
    plotting_bounds: tuple,
    alpha: float = 0.5,
    marker="o",
    color=None,
    marker_size=10,
):
    """
    Draw 2D sample plot.
    This function returns scatter object.
    """
    x_lower_lim, x_upper_lim, y_lower_lim, y_upper_lim = plotting_bounds

    sample = sample.cpu().detach()
    x = torch.clamp(sample[:, 0], x_lower_lim, x_upper_lim)
    y = torch.clamp(sample[:, 1], y_lower_lim, y_upper_lim)
    return ax.scatter(x, y, alpha=alpha, marker=marker, s=marker_size, c=color)


def draw_2D_kde(sample: torch.Tensor, ax: Axes, plotting_bounds: tuple):
    sample = sample.cpu().detach()
    try:
        return sns.kdeplot(
            x=sample[:, 0],
            y=sample[:, 1],
            cmap="Blues",
            fill=True,
            ax=ax,
            clip=plotting_bounds,
            warn_singular=False,
        )
    except ValueError as e:
        # If the KDE plot fails, return None
        print(f"Error in kdeplot: {e}")
        return None


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
    log_reward = torch.clamp(log_reward, min=range[0], max=range[1])

    hist, bins = np.histogram(
        log_reward.detach().cpu().numpy(), bins=bins, range=range, density=True
    )

    ax.set_xlabel("log reward")
    ax.set_ylabel("count")
    ax.grid(True)

    return ax.plot(bins[1:], hist, label=name, linewidth=3)


def draw_interatomic_dist_histogram(
    ax: Axes, interatomic_dist: torch.Tensor, name=None, bins=40, range=(0, 3)
):
    interatomic_dist = torch.clamp(interatomic_dist, min=range[0], max=range[1])

    hist, bins = np.histogram(
        interatomic_dist.detach().cpu().numpy(), bins=bins, range=range, density=True
    )

    ax.set_xlabel("Interatomic Distances")
    ax.set_ylabel("Normalized density")
    ax.grid(True)

    return ax.plot(bins[1:], hist, label=name, linewidth=3)


def draw_vector_field(
    ax: Axes,
    vecfield: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    plotting_bounds: tuple,
    grid_width: int,
):
    """
    Draw vector field quiver plot on 2D plot.
    Here, given vf must be batch-support version.
    """

    X, Y = make_2D_meshgrid(plotting_bounds, grid_width)

    points = get_points_on_2D_grid(
        plotting_bounds,
        grid_width,
        device=device,
    )

    vectors = vecfield(points).detach()

    return ax.quiver(X, Y, vectors[:, 0], vectors[:, 1])


def draw_trajectory_plot(
    ax: Axes,
    trajectory: torch.Tensor,
):
    trajectory = trajectory.cpu().numpy()

    # extract x, y coordinates.
    traj_xs = trajectory[..., 0]
    traj_ys = trajectory[..., 1]

    # Plot each trajectory (maximum ten trajectory will be plotted).
    for traj_x, traj_y, _ in zip(traj_xs, traj_ys, range(0, 10)):
        ax.plot(traj_x, traj_y, color="black", clip_on=True)

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
