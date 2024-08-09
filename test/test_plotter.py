import torch

from energy import Plotter, GMM25, GMM9, Funnel, ManyWell


FIGURE_PATH = "./test/figure/"


def test_plotter_make_sample_plot_for_gmm25():
    energy_function = GMM25(device="cuda:0", dim=2)
    plotter = Plotter(
        energy_function=energy_function, fig_size=(12, 7), plotting_bounds=(-20, 20)
    )

    sample = torch.randn((100, 2), device="cuda:0")

    fig, _ = plotter.make_sample_plot(sample)
    fig.savefig(FIGURE_PATH + "gmm25_test_figure.png")


def test_plotter_make_sample_plot_for_gmm9():
    energy_function = GMM9(device="cuda:0", dim=2)
    plotter = Plotter(energy_function=energy_function, fig_size=(12, 7))

    sample = torch.randn((100, 2), device="cuda:0")

    fig, _ = plotter.make_sample_plot(sample)
    fig.savefig(FIGURE_PATH + "gmm9_test_figure.png")


def test_plotter_make_sample_plot_for_manywell():
    energy_function = ManyWell(device="cuda:0", dim=32)

    # Ground truth sample plot
    sample = energy_function.sample(1000)

    plotter = Plotter(
        energy_function=energy_function,
        fig_size=(12, 6),
        projection_dims=[(0, 2), (1, 2)],
        plotting_bounds=(-3, 3),
    )

    fig, _ = plotter.make_sample_plot(sample)
    fig.savefig(FIGURE_PATH + "manywell_test_figure.png")


def test_plotter_make_sample_plot_for_funnel():
    energy_function = Funnel(device="cuda:0", dim=3)

    # Ground truth sample plot
    sample = energy_function.sample(1000)

    plotter = Plotter(
        energy_function=energy_function,
        fig_size=(12, 6),
        projection_dims=[(0, 1), (1, 2)],
    )

    fig, _ = plotter.make_sample_plot(sample)
    fig.savefig(FIGURE_PATH + "funnel_test_figure.png")


def test_plotter_make_traj_plot_for_gmm25():
    energy_function = GMM25(device="cuda:0", dim=2)
    plotter = Plotter(
        energy_function=energy_function, fig_size=(12, 7), plotting_bounds=(-16, 16)
    )

    # Brownian motion trajectory
    trajectory = torch.randn((10, 10, 2), device="cuda:0") * 0.1
    trajectory = trajectory.cumsum(dim=-1)

    animation, _, _ = plotter.make_sample_generation_animation(trajectory)

    animation.save(FIGURE_PATH + "gmm25_test_traj_animation.gif")


def test_plotter_make_traj_plot_for_manywell():
    energy_function = ManyWell(device="cuda:0", dim=32)

    plotter = Plotter(
        energy_function=energy_function,
        fig_size=(12, 6),
        projection_dims=[(0, 2), (1, 2)],
        plotting_bounds=(-3, 3),
    )

    # Brownian motion trajectory
    trajectory = torch.randn((10, 10, 32), device="cuda:0") * 0.1
    trajectory = trajectory.cumsum(dim=-1)

    animation, *_ = plotter.make_sample_generation_animation(trajectory, 0, 1)
    animation.save(FIGURE_PATH + "manywell_test_traj_animation.gif")
