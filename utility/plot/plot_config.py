PLOT_CONFIG = {
    "Funnel": {
        "plotting_bounds": (-10.0, 10.0),
        "projection_dims": [(0, 1), (1, 2)],
        "fig_size": (12.0, 6.0),
        "sample-plot": True,
        "kde-plot": True,
        "energy-hist": False,
    },
    "ManyWell": {
        "plotting_bounds": (-3.0, 3.0),
        "projection_dims": [(0, 2), (1, 2)],
        "fig_size": (12.0, 6.0),
        "energy_range": (90, 160),
        "sample-plot": True,
        "kde-plot": True,
        "energy-hist": True,
    },
    "ManyWell128": {
        "plotting_bounds": (-3.0, 3.0),
        "projection_dims": [(0, 2), (1, 2)],
        "fig_size": (12.0, 6.0),
        "energy_range": (450, 580),
        "sample-plot": True,
        "kde-plot": True,
        "energy-hist": True,
    },
    "GMM9": {
        "sample-plot": True,
        "kde-plot": True,
        "energy-hist": False,
        "plotting_bounds": (-10.0, 10.0),
        "fig_size": (12.0, 12.0),
    },
    "GMM25": {
        "sample-plot": True,
        "kde-plot": True,
        "energy-hist": False,
        "plotting_bounds": (-20.0, 20.0),
        "fig_size": (12.0, 12.0),
    },
    "GMM40": {
        "sample-plot": True,
        "kde-plot": True,
        "energy-hist": False,
        "plotting_bounds": (-50.0, 50.0),
        "fig_size": (12.0, 12.0),
    },
    # "DW4": {
    # },
    # "LJ13": {
    # },
    # "LJ55": {
    # },
    "LGCP": {
        "sample-plot": False,
        "kde-plot": False,
        "energy-hist": True,
        "energy_range": (0, 10),
        "fig_size": (12.0, 7.0),
    },
}
