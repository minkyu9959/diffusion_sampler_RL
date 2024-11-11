from omegaconf import OmegaConf

FUNNEL_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "plotting_bounds": (-15.0, 15.0, -80.0, 80.0),
        "func_min_value": -50.0,
        "n_contour_levels": 10,
        "fill_color": True,
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "marker": "x",
        "marker_size": 10,
        "color": "Red",
        "alpha": 0.5,
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "sample_figure": {
        "fig_size": (5.0, 3.0),
        "projection_dims": [(0, 1)],
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
}

MANYWELL_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        "fill_color": False,
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "sample_figure": {
        "fig_size": (12.0, 6.0),
        "projection_dims": [(0, 2), (1, 2)],
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (90, 160),
    },
}

MANYWELL128_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "sample_figure": {
        "projection_dims": [(0, 2), (1, 2)],
        "fig_size": (12.0, 6.0),
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (450, 580),
    },
}

MANYWELL512_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "sample_figure": {
        "projection_dims": [(0, 2), (1, 2), (2, 4), (3, 4)],
        "fig_size": (12.0, 12.0),
    },
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (1000, 2300),
    },
}

SYMMETRIC_MANYWELL_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "plotting_bounds": (-3.0, 3.0, -3.0, 3.0),
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "sample_figure": {
        "projection_dims": [(0, 2), (1, 2)],
        "fig_size": (12.0, 6.0),
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (90, 160),
    },
}

GMM9_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
        "plotting_bounds": (-10.0, 10.0, -10.0, 10.0),
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "sample_figure": {
        "fig_size": (12.0, 12.0),
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
}

GMM25_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
        "plotting_bounds": (-20.0, 20.0, -20.0, 20.0),
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "sample_figure": {
        "fig_size": (12.0, 12.0),
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
}

GMM40_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
        "plotting_bounds": (-50.0, 50.0, -50.0, 50.0),
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "sample_figure": {
        "fig_size": (12.0, 12.0),
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
}

MoG50_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
        "plotting_bounds": (-20.0, 20.0, -20.0, 20.0),
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "sample_figure": {
        "fig_size": (12, 12),
        "projection_dims": [(0, 1), (0, 2), (1, 2), (1, 3)],
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
}

MoG200_CONFIG = {
    "contour_plot": {
        "grid_width": 200,
        "n_contour_levels": 50,
        "func_min_value": -1000.0,
        "fill_color": False,
        "plotting_bounds": (-20.0, 20.0, -20.0, 20.0),
    },
    "sample_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
        "alpha": 0.4,
    },
    "sample_figure": {
        "fig_size": (12, 12),
        "projection_dims": [(0, 1), (0, 2), (1, 2), (1, 3)],
    },
    "kde_plot": {
        "plotting_bounds": "${contour_plot.plotting_bounds}",
    },
    "kde_figure": {
        "fig_size": "${sample_figure.fig_size}",
        "projection_dims": "${oc.select:sample_figure.projection_dims,null}",
    },
}

DW4_CONFIG = {
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (10, 30),
    },
    "interatomic_dist": {
        "range": (1, 7),
        "bins": 100,
    },
}

LJ13_CONFIG = {
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (-10, 70),
    },
    "interatomic_dist": {
        "range": (0, 6),
        "bins": 100,
    },
}

LJ55_CONFIG = {
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (200, 400),
    },
    "interatomic_dist": {
        "range": (0, 6),
        "bins": 100,
    },
}

LGCP_CONFIG = {
    "energy_hist": {
        "fig_size": (8, 6),
        "bins": 40,
        "energy_range": (0, 1000),
    },
}

ALDP_CONFIG = {
    "interatomic_dist": {
        "range": (-0.1, 2.0),
        "bins": 100,
    },
}

CONFIG_FOR_ENERGY = {
    "Funnel": FUNNEL_CONFIG,
    "ManyWell": MANYWELL_CONFIG,
    "SymmetricManyWell": SYMMETRIC_MANYWELL_CONFIG,
    "ManyWell128": MANYWELL128_CONFIG,
    "ManyWell512": MANYWELL512_CONFIG,
    "GMM9": GMM9_CONFIG,
    "GMM25": GMM25_CONFIG,
    "GMM40": GMM40_CONFIG,
    "MoG50": MoG50_CONFIG,
    "MoG200": MoG200_CONFIG,
    "DW4": DW4_CONFIG,
    "LJ13": LJ13_CONFIG,
    "LJ55": LJ55_CONFIG,
    "LGCP": LGCP_CONFIG,
    "AlanineDipeptide": ALDP_CONFIG,
}

CONFIG_FOR_ENERGY = CONFIG_FOR_ENERGY
