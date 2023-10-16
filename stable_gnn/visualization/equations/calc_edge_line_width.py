import numpy as np

from stable_gnn.visualization.config.parameters.defaults import Defaults


def calculate_edge_line_width(edges_list):
    return Defaults.edge_line_width_multiplier * np.exp(-len(edges_list) / Defaults.edge_line_width_divider)
