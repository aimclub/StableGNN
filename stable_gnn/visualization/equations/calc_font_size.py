import numpy as np

from stable_gnn.visualization.config.parameters.defaults import Defaults


def calculate_font_size(vertex_num):
    return Defaults.font_size_multiplier * np.exp(-vertex_num / Defaults.font_size_divider)
