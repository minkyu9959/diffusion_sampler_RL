from omegaconf import OmegaConf

DEFAULT_CONFIG = {
    "contour_plot": {
        "grid_width_n_points": 200,
        "n_contour_levels": 50,
        "log_prob_min": -1000.0,
        "fill_color": False,
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
    },
    "interatomic_dist": None,
}
DEFAULT_CONFIG = OmegaConf.create(DEFAULT_CONFIG)


CONFIG_FOR_ENERGY = {
    "Funnel": {
        "contour_plot": {
            "plotting_bounds": (-15.0, 15.0, -80.0, 80.0),
            "log_prob_min": -50.0,
            "n_contour_levels": 10,
            "fill_color": True,
        },
        "sample_plot": {
            "marker": "x",
            "marker_size": 10,
            "color": "Red",
            "alpha": 0.5,
        },
        "sample_figure": {
            "fig_size": (5.0, 3.0),
            "projection_dims": [(0, 1)],
        },
        "energy_hist": None,
    },
    "ManyWell": {
        "contour_plot": {
            "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        },
        "sample_figure": {
            "fig_size": (12.0, 6.0),
            "projection_dims": [(0, 2), (1, 2)],
        },
        "energy_hist": {
            "energy_range": (90, 160),
        },
    },
    "SymmetricManyWell": {
        "contour_plot": {
            "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        },
        "sample_figure": {
            "fig_size": (12.0, 6.0),
            "projection_dims": [(0, 2), (1, 2)],
        },
        "energy_hist": {
            "energy_range": (90, 160),
        },
    },
    "ManyWell128": {
        "contour_plot": {
            "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        },
        "sample_figure": {
            "projection_dims": [(0, 2), (1, 2)],
            "fig_size": (12.0, 6.0),
        },
        "energy_hist": {
            "energy_range": (450, 580),
        },
    },
    "ManyWell512": {
        "contour_plot": {
            "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        },
        "sample_figure": {
            "projection_dims": [(0, 2), (1, 2), (2, 4), (3, 4)],
            "fig_size": (12.0, 12.0),
        },
        "kde_plot": None,
        "kde_figure": None,
        "energy_hist": {
            "energy_range": (1000, 2300),
            "bins": 400,
        },
    },
    "GMM9": {
        "contour_plot": {
            "plotting_bounds": (-10.0, 10.0, -10.0, 10.0),
        },
        "sample_figure": {
            "fig_size": (12.0, 12.0),
        },
        "energy_hist": None,
    },
    "GMM25": {
        "contour_plot": {
            "plotting_bounds": (-20.0, 20.0, -20.0, 20.0),
        },
        "sample_figure": {
            "fig_size": (12.0, 12.0),
        },
        "energy_hist": None,
    },
    "GMM40": {
        "contour_plot": {
            "plotting_bounds": (-50.0, 50.0, -50.0, 50.0),
        },
        "sample_figure": {
            "fig_size": (12.0, 12.0),
        },
        "energy_hist": None,
    },
    "MoG50": {
        "contour_plot": {
            "plotting_bounds": (-0.2, 0.2, -0.2, 0.2),
        },
        "sample_figure": {
            "fig_size": (12, 12),
            "projection_dims": [(0, 1), (0, 2), (1, 2), (1, 3)],
        },
        "energy_hist": None,
    },
    "MoG200": {
        "contour_plot": {
            "plotting_bounds": (-0.2, 0.2, -0.2, 0.2),
        },
        "sample_figure": {
            "fig_size": (12, 12),
            "projection_dims": [(0, 1), (0, 2), (1, 2), (1, 3)],
        },
        "energy_hist": None,
    },
    "DW4": {
        "contour_plot": None,
        "sample_plot": None,
        "kde_plot": None,
        "sample_figure": None,
        "kde_figure": None,
        "energy_hist": {
            "energy_range": (10, 30),
        },
        "interatomic_dist": {
            "range": (1, 7),
            "bins": 100,
        },
    },
    "LJ13": {
        "contour_plot": None,
        "sample_plot": None,
        "kde_plot": None,
        "sample_figure": None,
        "kde_figure": None,
        "energy_hist": {
            "energy_range": (-10, 70),
        },
        "interatomic_dist": {
            "range": (0, 6),
            "bins": 100,
        },
    },
    "LJ55": {
        "contour_plot": None,
        "sample_plot": None,
        "kde_plot": None,
        "sample_figure": None,
        "kde_figure": None,
        "energy_hist": {
            "energy_range": (200, 400),
        },
        "interatomic_dist": {
            "range": (0, 6),
            "bins": 100,
        },
    },
    "LGCP": {
        "contour_plot": None,
        "sample_plot": None,
        "kde_plot": None,
        "sample_figure": None,
        "kde_figure": None,
        "energy_hist": {
            "energy_range": (0, 1000),
        },
    },
}

CONFIG_FOR_ENERGY = OmegaConf.create(CONFIG_FOR_ENERGY)
