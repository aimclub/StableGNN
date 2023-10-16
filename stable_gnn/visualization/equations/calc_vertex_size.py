import numpy as np

from stable_gnn.visualization.config.parameters.defaults import Defaults


def calculate_vertex_size(vertex_num):
    return (Defaults.calculate_vertex_size_multiplier /
            np.sqrt(vertex_num + Defaults.calculate_vertex_size_divider) *
            Defaults.calculate_vertex_size_modifier)
